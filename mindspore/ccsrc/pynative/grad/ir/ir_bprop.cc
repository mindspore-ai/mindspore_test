/**
 * Copyright 2024 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "pynative/grad/ir/ir_bprop.h"
#include <string>
#include <vector>
#include <memory>
#include "pynative/pynative_utils.h"
#include "pynative/grad/grad_utils.h"
#include "include/common/utils/primitive_utils.h"
#include "include/common/pynative/common_utils.h"
#include "pipeline/jit/ps/pass.h"
#include "ir/func_graph_cloner.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "frontend/optimizer/ad/pynative_jit_grad.h"

namespace mindspore::pynative::autograd {
namespace {
constexpr size_t kOutAndDoutNum = 2;
const mindspore::HashSet<std::string> kMonadOp = {kLoadOpName, kDependOpName, kUpdateStateOpName};
const mindspore::HashSet<std::string> kMetaFuncGraphOp{
  kPyExecuteOpName,
  kAttrMutableOpName,
  kMakeDictOpName,
};

void ClearGradMetaData(const ValuePtr &value) {
  if (value->isa<tensor::BaseTensor>()) {
    auto tensor = value->cast<tensor::BaseTensorPtr>();
    tensor->set_auto_grad_meta_data(nullptr);
  } else if (value->isa<ValueSequence>()) {
    auto value_sequence = value->cast<ValueSequencePtr>();
    for (const auto &val : value_sequence->value()) {
      ClearGradMetaData(val);
    }
  }
}

// Handle bprob of op which input dtype is real number and output dtype is complex number.
// If the dtype of a gradient(din) is complex number and the input of that is real number,
// only the real part of the gradient make sense in back propagate. So we handle it by
// insert a Real() ops after the gradient.
// input: AnfNode with input of op which input dtype is real number and output dtype is complex number.
// din: CNodePtr with gradient of input.
// tape: Funcgraph witch input and din belong to.
// return: New din with inserted real op if necessarily.
AnfNodePtr HandleRealToComplex(const tensor::BaseTensorPtr &input, const AbstractBasePtr &abs, const AnfNodePtr &din,
                               const KernelGraphPtr &tape) {
  MS_EXCEPTION_IF_NULL(din);
  TypePtr din_type = din->Type();
  if (din_type == nullptr || !din_type->isa<TensorType>()) {
    return din;
  }
  din_type = din_type->cast_ptr<TensorType>()->element();
  MS_EXCEPTION_IF_NULL(din_type);
  // cppcheck-suppress unreadVariable
  if (MS_LIKELY(din_type->type_id() != kNumberTypeComplex64 && din_type->type_id() != kNumberTypeComplex128)) {
    return din;
  }

  MS_EXCEPTION_IF_NULL(input);
  TypePtr input_type = input->Dtype();
  if (input_type == nullptr) {
    return din;
  }
  if (input_type->type_id() == kNumberTypeComplex64 || input_type->type_id() == kNumberTypeComplex128) {
    return din;
  }

  AnfNodePtr new_din = tape->FuncGraph::NewCNode({NewValueNode(prim::kPrimReal), din});
  AbstractBasePtr real_abs =
    std::make_shared<abstract::AbstractTensor>(abstract::AbstractTensor(input_type, abs->GetShapeTrack()));
  new_din->set_abstract(real_abs);
  return new_din;
}
}  // namespace

void ClearAutoGradCache() {
  mindspore::ad::ClearGradCache();
  bprop_pass::ClearCache();
  PyNativeAlgo::AutoGradUtil::ClearAutoGradStaticCache();
}

void IrBprop::BuildCustomBpropCNode(const CNodePtr &cnode, const PrimitivePtr &prim, std::vector<CNodePtr> *outputs) {
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(DEBUG) << "Try build custom bprop: " << prim->name();
  {
    py::gil_scoped_acquire gil;
    auto prim_py = prim->cast<PrimitivePyPtr>();
    if (prim_py == nullptr) {
      MS_LOG(DEBUG) << "Prim is not PrimitivePy, can not find python bprop";
      return;
    }
    py::function fn = prim_py->GetBpropFunction();
    if (py::isinstance<py::none>(fn)) {
      fn = GetBpropFunction(prim->name());
    }
    if (!fn || py::isinstance<py::none>(fn)) {
      MS_LOG(INFO) << "Can not find bprop function for " << prim->name() << ". fn: " << ConvertPyObjToString(fn);
      return;
    }
    (void)prim_py->SetHookFn(fn, HookType::kCustomOpBprop);
  }
  BuildBPropCutCNode(cnode, prim, outputs);
}

void IrBprop::BuildBPropCutCNode(const CNodePtr &cnode, const PrimitivePtr &prim, std::vector<CNodePtr> *outputs,
                                 size_t weight_size, bool is_need_recompute) {
  MS_EXCEPTION_IF_NULL(prim);
  auto bprop_cut = PyNativeAlgo::AutoGradUtil::BuildBpropCutPrim(prim, is_need_recompute);
  size_t origin_input_size = cnode->size() - kOutAndDoutNum - weight_size;
  // Create gradient outputs cnode
  AnfNodePtrList inputs{NewValueNode(bprop_cut)};
  if (PyNativeAlgo::Common::IsHookNeedSaveInputs(bprop_cut)) {
    for (size_t i = 1; i < origin_input_size; ++i) {
      (void)inputs.emplace_back(cnode->input(i));
    }
    if (!is_need_recompute) {
      // If not recompute, we should add out as bprop input.
      (void)inputs.emplace_back(cnode->input(cnode->size() - kOutAndDoutNum));
    }
  }
  (void)inputs.emplace_back(cnode->input(cnode->size() - 1));
  auto bprop_cut_cnode = ad_param_->tape_->FuncGraph::NewCNode(inputs);
  AbstractBasePtrList abs_list;
  // Only add last input dout to user.
  AddUser(cnode->input(cnode->size() - 1), bprop_cut_cnode, bprop_cut_cnode->size() - 1);
  for (size_t i = 1; i < cnode->size() - kOutAndDoutNum; ++i) {
    // Input may be parameter, we need add to user map.
    if (i < origin_input_size && PyNativeAlgo::Common::IsHookNeedSaveInputs(bprop_cut)) {
      AddUser(cnode->input(i), bprop_cut_cnode, i);
    }
    auto din = ad_param_->tape_->FuncGraph::NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), bprop_cut_cnode, NewValueNode(static_cast<int64_t>(i - 1))});
    MS_EXCEPTION_IF_NULL(cnode->input(i)->abstract());
    din->set_abstract(cnode->input(i)->abstract());
    (void)abs_list.emplace_back(cnode->input(i)->abstract());
    (void)outputs->emplace_back(din);
  }
  bprop_cut_cnode->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
  ad_param_->tape_->set_flag(kFlagPyNativeBpropGraphWithBpropCut, true);
  bprop_graph_run_by_single_op_ = true;
}

AnfNodePtr IrBprop::MapParameter(const ValuePtr &value, const abstract::AbstractBasePtr &abs,
                                 MetaGradInfoList *param_meta_grad_info) {
  if (value->isa<tensor::BaseTensor>()) {
    const auto &tensor = value->cast<tensor::BaseTensorPtr>();
    const auto &auto_grad_meta_data = impl::get_autograd_meta_impl(tensor);
    if (auto_grad_meta_data == nullptr) {
      MS_LOG(DEBUG) << "The tensor is a constant value, not a parameter!";
      return PyNativeAlgo::Common::CreateValueNodeByValue(tensor, abs);
    }
    const auto &param = auto_grad_meta_data->parameter();
    if (param != nullptr) {
      // In dynamic shape scenario, abs my be need change
      param->set_abstract(abs);
      return param;
    }
    (*param_meta_grad_info)[tensor] = auto_grad_meta_data;
    set_bprop_graph_run_by_single_op(auto_grad_meta_data->is_register_hook());
    if (auto_grad_meta_data->input_type() == InputType::kParameter &&
        PyNativeAlgo::AutoGradUtil::IsParamRequiresGrad(tensor)) {
      return AddParameterNode(tensor, abs);
    }
    return PyNativeAlgo::Common::CreateValueNodeByValue(value, abs);
  }
  if (value->isa<ValueSequence>()) {
    const auto &val_seq = value->cast<ValueSequencePtr>()->value();
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (val_seq.size() != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Get value sequence size " << val_seq.size() << " not equal to abstract size "
                        << abs_seq->size();
    }
    AnfNodePtrList inputs;
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t i = 0; i < val_seq.size(); ++i) {
      (void)inputs.emplace_back(MapParameter(val_seq[i], abs_seq->elements()[i], param_meta_grad_info));
    }
    auto cnode = ad_param_->tape_->FuncGraph::NewCNode(inputs);
    // For replacing fg parameter by user
    for (size_t i = 1; i < inputs.size(); ++i) {
      AddUser(inputs[i], cnode, i);
    }
    cnode->set_abstract(abs);
    return cnode;
  }
  if (value->isa<tensor::COOTensor>()) {
    const auto &coo_tensor = value->cast<tensor::COOTensorPtr>();
    return MapParameter(coo_tensor->GetIndices(), abs, param_meta_grad_info);
  }

  if (value->isa<tensor::CSRTensor>()) {
    const auto &csr_tensor = value->cast<tensor::CSRTensorPtr>();
    return MapParameter(csr_tensor->GetIndices(), abs, param_meta_grad_info);
  }
  return PyNativeAlgo::Common::CreateValueNodeByValue(value, abs);
}

ParameterPtr IrBprop::AddParameterNode(const tensor::BaseTensorPtr &tensor, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto param = CreateTapeParameter(tensor, abs);
  auto zeros_like_dout = PyNativeAlgo::AutoGradUtil::BuildSpecialNode(
    ad_param_->tape_, PyNativeAlgo::AutoGradUtil::GetFakeZeroTensor(), param->abstract(), SpecialType::kZerosLikeType);
  auto func_node = std::make_shared<IrFunctionNode>(ad_param_->tape_, zeros_like_dout);
  auto input_adjoint = std::make_shared<IrVariable>(func_node, tensor, true);
  (void)ad_param_->variable_adjoint_set_.insert(input_adjoint);
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
  auto_grad_meta_data->set_variable(input_adjoint);
  (void)ad_param_->weights_used_in_graph_.emplace_back(param);
  return param;
}

ParameterPtr IrBprop::CreateTapeParameter(const tensor::BaseTensorPtr &tensor, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(abs);
  auto param = ad_param_->fg_->add_parameter();
  param->set_abstract(abs);
  if (tensor->is_parameter()) {
    param->set_default_param(tensor);
  }
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  if (auto_grad_meta_data == nullptr) {
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    tensor->set_auto_grad_meta_data(auto_grad_meta_data);
  }
  auto_grad_meta_data->set_input_type(InputType::kParameter);
  auto_grad_meta_data->set_parameter(param);
  return param;
}

void IrBprop::UpdateNextEdges(const VariablePtr &variable, const std::vector<CNodePtr> &dins,
                              const ValuePtrList &inputs_value, const abstract::AbstractBasePtrList &abs,
                              const string &op_name) {
  size_t input_size = inputs_value.size();
  if (dins.size() != input_size) {
    MS_LOG(EXCEPTION) << "The size of dins " << dins.size() << " is not same as input_value " << input_size;
  }
  const auto &fn = variable->ir_function_node();
  for (size_t i = 0; i < input_size; ++i) {
    auto din = dins[i];
    MS_EXCEPTION_IF_NULL(din);
    MS_LOG(DEBUG) << "Input arg id: " << PyNativeAlgo::Common::GetIdByValue(inputs_value[i]) << ", din "
                  << din->DebugString();
#ifndef ENABLE_TEST
    // VM no need run pass
    din = pass_forward_->PassForDin(din, op_name, false);
#endif
    UpdateNextEdge(fn, din, inputs_value[i], abs[i]);
  }
  if (fn->next_edges().empty()) {
    variable->set_is_need_grad(false);
  }
  MS_LOG(DEBUG) << "Finish update next edges for variable: " << variable->ToString();
}

void IrBprop::AddUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  MS_EXCEPTION_IF_NULL(ad_param_);
  (void)ad_param_->users_.dout_user_[node].emplace_back(user, index);
}

void IrBprop::AddReverseUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  (void)ad_param_->reverse_users_[node].emplace_back(user, index);
}

void IrBprop::BackPropagate() {
  UpdateLazyUser();
  const auto &last_node_reverse_iter = GetLastNodeReverseIter();
#ifndef ENABLE_TEST
  SeenNum seen = NewSeenGeneration();
#endif
  MS_LOG(DEBUG) << "Is running recompute grad " << is_run_recompute_;
  for (auto iter = last_node_reverse_iter; iter != ad_param_->variable_adjoint_set_.rend(); ++iter) {
    const auto &variable = *iter;
    const auto &fn = variable->ir_function_node();
    if (!variable->is_need_propagate() || !variable->is_need_grad()) {
      MS_LOG(DEBUG) << "No need grad, variable is: " << variable->ToString();
      LeafNodeButHasTensorHook(variable, fn);
      continue;
    }
    if (static_cast<bool>(MS_UNLIKELY(variable->is_fake_bprop()))) {
      MS_LOG(EXCEPTION) << "Illegal primitive " << variable->fake_prim_name() << "'s bprop not defined";
    }
    MS_LOG(DEBUG) << "Begin backpropagate: " << variable->ToString();
    // If zeroslike not used in funcgraph, we need replace the zeroslike placeholder with real zeroslike value.
    if (static_cast<bool>(MS_UNLIKELY(PyNativeAlgo::AutoGradUtil::IsZerosLikeNode(fn->accumulate_dout())))) {
      fn->set_accumulate_dout(PyNativeAlgo::AutoGradUtil::BuildSpecialNode(
        fn->tape(), variable->out_value(), fn->accumulate_dout()->abstract(), SpecialType::kZerosLikeType));
    }
    // If register hook by weight, and weight in recomputed cell.So, hook will execute, which is not expected.
    if (!is_run_recompute_ || !variable->is_leaf()) {
      fn->set_accumulate_dout(pass_forward_->PassBackwardHook(variable->out_value(), fn->accumulate_dout()));
    }
    // Replace real dout to fake dout, update replace result to eliminate tuplegetitem
    // when accumulate_dout is tuplegetitem
    Replace(fn->fake_dout(), fn->accumulate_dout(), &ad_param_->users_.dout_user_, true);
    // replace edges which exist fake dout
    fn->ReplaceEdges();
    const auto &next_edges = fn->next_edges();
    for (const auto &next_edge : next_edges) {
      const auto &last_variable = next_edge.first;
      const auto &din = next_edge.second;
      // If din is Zeroslike, It represents that this tensor grad is zeros.
      if (static_cast<bool>(MS_UNLIKELY(PyNativeAlgo::AutoGradUtil::IsZerosLikeNode(din)))) {
        if (!last_variable->is_custom_op_variable()) {
          MS_LOG(DEBUG) << last_variable->ToString() << ", its gradient is zeroslike, no need propagate!";
          continue;
        }
        MS_LOG(DEBUG) << "Get custom bprop variable, zeros input din may be have non zeors dout";
      }
#ifndef ENABLE_TEST
      // VM no need run pass
      pass_forward_->ConvertMakeTupleInputToDynamicInput(din, seen, bprop_graph_run_by_single_op_);
#endif
      last_variable->ir_function_node()->UpdateAccumulativeDout(din);
      last_variable->set_is_need_propagate(true);
    }
  }
  MS_LOG(DEBUG) << "End BackPropagate";
}

void IrBprop::LeafNodeButHasTensorHook(const IrVariablePtr &variable, const IrFunctionNodePtr &fn) const {
  if (!variable->is_leaf()) {
    return;
  }
  fn->set_accumulate_dout(pass_forward_->PassBackwardHook(variable->out_value(), fn->accumulate_dout()));
}

OrderedSet<IrVariablePtr>::reverse_iterator IrBprop::GetLastNodeReverseIter() const {
  for (auto iter = ad_param_->variable_adjoint_set_.rbegin(); iter != ad_param_->variable_adjoint_set_.rend(); ++iter) {
    if (*iter == ad_param_->last_variable_) {
      ad_param_->last_variable_->set_is_need_propagate(true);
      return iter;
    }
  }
  return ad_param_->variable_adjoint_set_.rend();
}

AbstractBasePtr IrBprop::BuildForwardLastNode(bool has_aux) {
  if (has_aux) {
    if (!ad_param_->sens_value_->isa<ValueSequence>()) {
      MS_LOG(EXCEPTION)
        << "If you set has aux for grad or value_and_grad, that forward function should be multi output, but got "
        << ad_param_->sens_value_->ToString();
    }
    ad_param_->sens_value_ = ad_param_->sens_value_->cast<ValueSequencePtr>()->value()[0];
  }
  MS_LOG(DEBUG) << "Process last node info " << PyNativeAlgo::Common::GetIdByValue(ad_param_->sens_value_);
  auto zeros_like_node = PyNativeAlgo::AutoGradUtil::BuildSpecialNode(ad_param_->tape_, ad_param_->sens_value_, nullptr,
                                                                      SpecialType::kZerosLikeType);
  auto fn = std::make_shared<IrFunctionNode>(ad_param_->tape_, zeros_like_node);
  auto sens_variable = std::make_shared<IrVariable>(fn, ad_param_->sens_value_);
  if (ad_param_->sens_value_->isa<tensor::BaseTensor>()) {
    const auto &sens_tensor = ad_param_->sens_value_->cast<tensor::BaseTensorPtr>();
    if (const auto &auto_grad_meta_data = sens_tensor->auto_grad_meta_data();
        auto_grad_meta_data == nullptr || PyNativeAlgo::Common::IsConstant(auto_grad_meta_data->input_type())) {
      sens_variable->set_is_need_grad(false);
    }
  }
  UpdateNextEdge(fn, zeros_like_node, ad_param_->sens_value_, fn->accumulate_dout()->abstract());
  (void)ad_param_->variable_adjoint_set_.insert(sens_variable);
  ad_param_->last_variable_ = sens_variable;
  return fn->accumulate_dout()->abstract();
}

void IrBprop::Replace(const AnfNodePtr &old_node, const AnfNodePtr &new_node, expander::bprop::UserType *user,
                      bool need_update) {
  MS_EXCEPTION_IF_NULL(user);
  if (user->find(old_node) == user->end()) {
    return;
  }
  const auto &old_node_users = (*user)[old_node];
  for (const auto &pair_node : old_node_users) {
    auto cnode = pair_node.first.lock();
    if (cnode == nullptr) {
      continue;
    }
    size_t index = pair_node.second;
    if (index >= cnode->size()) {
      // After convert attr cnode input will less
      if (auto v = cnode->GetAttr(kAttrConvertAttrNode); v != nullptr) {
        index -= GetValue<size_t>(v);
      } else {
        MS_LOG(EXCEPTION) << "exception for index: " << index << "greater than cnode size: " << cnode->size();
      }
    }
    cnode->set_input(index, new_node);
    if (need_update && IsPrimitiveCNode(new_node, prim::kPrimTupleGetItem)) {
      AddTupleGetItemUser(new_node, cnode, index);
    }
  }
}

ValuePtrList IrBprop::GetInputArgs(const CNodePtr &cnode, AnfNodePtrList *cnode_inputs) const {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(cnode_inputs);
  ValuePtrList input_value;
  for (size_t i = 1; i < cnode->size(); ++i) {
    const auto &input_node = cnode->input(i);
    // Find knode and out value
    const auto it = ad_param_->anfnode_to_variable_adjoint_.find(input_node);
    if (it != ad_param_->anfnode_to_variable_adjoint_.end()) {
      (void)cnode_inputs->emplace_back(it->second->k_node());
      (void)input_value.emplace_back(it->second->out_value());
      continue;
    }
    if (input_node->isa<ValueNode>()) {
      auto v_node = input_node->cast<ValueNodePtr>();
      auto v = v_node->value();
      if (v != nullptr && v->isa<tensor::BaseTensor>()) {
        const auto &t = v->cast<tensor::BaseTensorPtr>();
        const auto &grad_meta = t->auto_grad_meta_data();
        // Jit forward graph has no parameters(input is tuple or constant), so input used in graph as valuenode, but it
        // is used by tape_ as parameter also
        if (grad_meta != nullptr && PyNativeAlgo::AutoGradUtil::IsParam(grad_meta->input_type())) {
          auto new_tensor = std::make_shared<tensor::Tensor>(t->data_type(), t->shape(), t->data_ptr());
          new_tensor->set_device_address(t->device_address());
          v = new_tensor;
        }
      }
      (void)PyNativeAlgo::AutoGradUtil::SetValueGradInfo(v, InputType::kConstant);
      // In case of jit forward graph and pynative bprop graph used same valuenode
      auto new_v_node = PyNativeAlgo::Common::CreateValueNodeByValue(v, v_node->abstract());
      (void)cnode_inputs->emplace_back(new_v_node);
      (void)input_value.emplace_back(v);
    } else {
      // Make Fake value
      auto v = MakeValue<int64_t>(0);
      (void)cnode_inputs->emplace_back(PyNativeAlgo::Common::CreateValueNodeByValue(v, input_node->abstract()));
      (void)input_value.emplace_back(v);
      MS_LOG(DEBUG) << "Get input node " << input_node->DebugString();
    }
  }
  return input_value;
}

AnfNodePtr IrBprop::BuildKNodeForMakeTuple(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_LOG(DEBUG) << "Build knode for MakeTuple " << input_node->DebugString();
  const auto &cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  AnfNodePtrList inputs{NewValueNode(prim::kPrimMakeTuple)};
  ValuePtrList input_value;
  AbstractBasePtrList input_abs;
  for (size_t i = 1; i < cnode->size(); ++i) {
    (void)inputs.emplace_back(BuildKNodeForCNodeInput(cnode->input(i)));
    if (cnode->input(i)->isa<CNode>() || cnode->input(i)->isa<Parameter>()) {
      const auto input_adjoint_iter = ad_param_->anfnode_to_variable_adjoint_.find(cnode->input(i));
      if (input_adjoint_iter == ad_param_->anfnode_to_variable_adjoint_.end()) {
        MS_LOG(EXCEPTION) << "Cannot find input in adjoint map, inp: " << cnode->input(i)->DebugString();
      }
      (void)input_value.emplace_back(input_adjoint_iter->second->out_value());
      (void)input_abs.emplace_back(cnode->input(i)->abstract());
    } else {
      auto value_node = cnode->input(i)->cast<ValueNodePtr>();
      MS_EXCEPTION_IF_NULL(value_node);
      (void)input_value.emplace_back(value_node->value());
      (void)input_abs.emplace_back(value_node->abstract());
    }
  }
  auto out_value = MakeValue(input_value);
  AnfNodePtr dout = PyNativeAlgo::AutoGradUtil::BuildSpecialNode(ad_param_->tape_, out_value, input_node->abstract(),
                                                                 SpecialType::kZerosLikeType);
  auto fn = std::make_shared<IrFunctionNode>(ad_param_->tape_, dout);
  auto variable_adjoint = std::make_shared<IrVariable>(fn, out_value);
  auto k_node = ad_param_->tape_->FuncGraph::NewCNode(inputs);
  k_node->set_abstract(input_node->abstract());
  variable_adjoint->set_k_node(k_node);
  // Create dout for maketuple
  std::vector<CNodePtr> make_tuple_dout;
  for (size_t i = 1; i < cnode->size(); ++i) {
    auto d = ad_param_->tape_->FuncGraph::NewCNode(
      {NewValueNode(prim::kPrimTupleGetItem), dout, NewValueNode(SizeToLong(i - 1))});
    d->set_abstract(cnode->input(i)->abstract());
    (void)make_tuple_dout.emplace_back(d);
    AddUser(dout, d, 1);
  }
  UpdateNextEdges(variable_adjoint, make_tuple_dout, input_value, input_abs);
  (void)ad_param_->anfnode_to_variable_adjoint_.insert(std::make_pair(input_node, variable_adjoint));
  (void)ad_param_->variable_adjoint_set_.insert(variable_adjoint);
  return k_node;
}

AnfNodePtr IrBprop::BuildKNodeForCNodeInput(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  if (input_node->isa<CNode>()) {
    const auto input_adjoint_iter = ad_param_->anfnode_to_variable_adjoint_.find(input_node);
    if (input_adjoint_iter == ad_param_->anfnode_to_variable_adjoint_.end()) {
      if (IsPrimitiveCNode(input_node, prim::kPrimMakeTuple)) {
        return BuildKNodeForMakeTuple(input_node);
      }
      if (IsPrimitiveCNode(input_node, prim::kPrimTupleGetItem)) {
        return BuildKNodeForTupleGetItem(input_node);
      }
      MS_LOG(EXCEPTION) << "Can not find input in adjoint map, inp: " << input_node->DebugString();
    }
    return input_adjoint_iter->second->k_node();
  }
  // Tuple sens will come in
  if (input_node->isa<Parameter>()) {
    const auto input_adjoint_iter = ad_param_->anfnode_to_variable_adjoint_.find(input_node);
    if (input_adjoint_iter != ad_param_->anfnode_to_variable_adjoint_.end() &&
        input_adjoint_iter->second->k_node() != nullptr) {
      return input_adjoint_iter->second->k_node();
    }
  }
  return input_node;
}

AnfNodePtr IrBprop::BuildKNodeForTupleGetItem(const AnfNodePtr &input_node) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_LOG(DEBUG) << "Build knode for TupleGetItem " << input_node->DebugString();
  const auto &tuple_item_cnode = input_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(tuple_item_cnode);
  // Find make tuple or sens(tuple) node for get out value
  const auto input_adjoint_iter = ad_param_->anfnode_to_variable_adjoint_.find(tuple_item_cnode->input(kIndex1));
  if (input_adjoint_iter == ad_param_->anfnode_to_variable_adjoint_.end()) {
    MS_LOG(EXCEPTION) << "Cannot find input in adjoint map, inp: " << tuple_item_cnode->input(kIndex1)->DebugString();
  }
  const auto &v_tuple = input_adjoint_iter->second->out_value()->cast<ValueSequencePtr>();
  MS_EXCEPTION_IF_NULL(v_tuple);
  auto index_value = GetValueNode<Int64ImmPtr>(tuple_item_cnode->input(kIndex2));
  auto index_value_int = LongToSize(index_value->value());
  auto out_value = (*v_tuple)[index_value_int];
  MS_EXCEPTION_IF_NULL(out_value);
  AnfNodePtr dout = PyNativeAlgo::AutoGradUtil::BuildSpecialNode(ad_param_->tape_, out_value, input_node->abstract(),
                                                                 SpecialType::kZerosLikeType);
  auto fn = std::make_shared<IrFunctionNode>(ad_param_->tape_, dout);
  auto variable_adjoint = std::make_shared<IrVariable>(fn, out_value);

  AnfNodePtrList inputs{NewValueNode(prim::kPrimTupleGetItem)};
  // Get make tuple knode
  (void)inputs.emplace_back(BuildKNodeForCNodeInput(tuple_item_cnode->input(kIndex1)));
  // Get index knode
  (void)inputs.emplace_back(BuildKNodeForCNodeInput(tuple_item_cnode->input(kIndex2)));
  auto k_node = ad_param_->tape_->FuncGraph::NewCNode(inputs);
  k_node->set_abstract(input_node->abstract());
  variable_adjoint->set_k_node(k_node);
  // Create dout for tuplegetitem
  AnfNodePtrList tuple_getitem_dout{NewValueNode(prim::kPrimMakeTuple)};
  const auto &abs_tuple = tuple_item_cnode->input(kIndex1)->abstract()->cast<abstract::AbstractSequencePtr>();
  for (size_t i = 0; i < v_tuple->size(); ++i) {
    const auto &v = v_tuple->value()[i];
    if (i == index_value_int) {
      (void)tuple_getitem_dout.emplace_back(dout);
    } else {
      (void)tuple_getitem_dout.emplace_back(PyNativeAlgo::AutoGradUtil::BuildSpecialNode(
        ad_param_->tape_, v, abs_tuple->elements()[i], SpecialType::kZerosLikeType));
    }
  }
  CNodePtr tuple_getitem_dout_value = ad_param_->tape_->FuncGraph::NewCNode(tuple_getitem_dout);
  tuple_getitem_dout_value->set_abstract(tuple_item_cnode->input(kIndex1)->abstract());
  auto index_dout_value =
    PyNativeAlgo::AutoGradUtil::BuildSpecialNode(
      ad_param_->tape_, index_value, tuple_item_cnode->input(kIndex1)->abstract(), SpecialType::kZerosLikeType)
      ->cast<CNodePtr>();
  UpdateNextEdges(variable_adjoint, {tuple_getitem_dout_value, index_dout_value}, {v_tuple, index_value},
                  {tuple_item_cnode->input(kIndex1)->abstract(), tuple_item_cnode->input(kIndex2)->abstract()});
  AddUser(dout, tuple_getitem_dout_value, index_value_int + 1);
  (void)ad_param_->anfnode_to_variable_adjoint_.insert(std::make_pair(input_node, variable_adjoint));
  (void)ad_param_->variable_adjoint_set_.insert(variable_adjoint);
  return k_node;
}

AnfNodePtr IrBprop::GetKnode(const PrimitivePtr &prim, const CNodePtr &cnode, const AnfNodePtrList &cnode_inputs,
                             bool jit_by_value) {
  if (IsPrimitiveEquals(prim, prim::kPrimMirror)) {
    return ad_param_->anfnode_to_variable_adjoint_.at(cnode->input(kIndex1))->k_node();
  }
  auto c_k_node = ad_param_->tape_->FuncGraph::NewCNode(cnode_inputs);
  c_k_node->set_abstract(cnode->abstract());
  // In jit, copy forward graph cnode info to bprop graph
  if (jit_by_value && cnode->forward().first != nullptr) {
    auto new_v_node =
      PyNativeAlgo::Common::CreateValueNodeByValue(cnode->forward().first->value(), cnode->forward().first->abstract());
    c_k_node->set_forward(new_v_node, cnode->forward().second);
    ad_param_->tape_->set_used_forward_nodes({c_k_node});
  }
  c_k_node->AddAttr(bprop_pass::kIsKNode, MakeValue(true));
  return c_k_node;
}

void IrBprop::UpdateNextEdgeForDict(const IrFunctionNodePtr &fn, const AnfNodePtr &din, const ValuePtr &input_arg,
                                    const AbstractBasePtr &abs) {
  auto value_dict = input_arg->cast<ValueDictionaryPtr>()->value();
  const auto &abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
  MS_EXCEPTION_IF_NULL(abs_dict);
  if (value_dict.size() != abs_dict->size()) {
    MS_LOG(EXCEPTION) << "Get value dict size " << value_dict.size() << " not equal to abstract size "
                      << abs_dict->size();
  }
  for (size_t i = 0; i < value_dict.size(); ++i) {
    auto sub_value = value_dict[i];
    auto key_item = PyNativeAlgo::Common::CreateValueNodeByValue(sub_value.first, abs_dict->elements()[i].first);
    CNodePtr new_din = ad_param_->tape_->FuncGraph::NewCNode({NewValueNode(prim::kPrimDictGetItem), din, key_item});
    new_din->set_abstract(CommonUtils::SetAbstractValueToAnyValue(abs_dict->elements()[i].second));
    if (din == fn->fake_dout()) {
      // The new_din's index input is fn->fake_dout()
      LazyAddUser(fn->fake_dout(), new_din, 1);
    }
    // Add next edge to fn
    UpdateNextEdge(fn, new_din, sub_value.second, abs_dict->elements()[i].second);
  }
}

void IrBprop::UpdateNextEdge(const IrFunctionNodePtr &fn, const AnfNodePtr &din, const ValuePtr &input_arg,
                             const AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(din);
  MS_EXCEPTION_IF_NULL(input_arg);
  if (input_arg->isa<tensor::BaseTensor>()) {
    tensor::BaseTensorPtr input_tensor = nullptr;
    input_tensor = input_arg->cast<tensor::BaseTensorPtr>();
    const auto &auto_grad_meta_data = input_tensor->auto_grad_meta_data();
    // Get scalar tensor
    if (auto_grad_meta_data == nullptr) {
      return;
    }
    auto variable = auto_grad_meta_data->UnsafeGetVariableImpl();
    if (variable == nullptr || !variable->is_need_grad()) {
      return;
    }
    auto real_din = HandleRealToComplex(input_tensor, abs, din, fn->tape());
    auto new_din = TraceInput(fn, variable->out_value(), variable->ir_function_node()->accumulate_dout()->abstract(),
                              input_tensor, real_din);
    fn->AddNextEdge(variable, new_din);
  } else if (input_arg->isa<ValueSequence>()) {
    auto value_seq = input_arg->cast<ValueSequencePtr>()->value();
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abs_seq);
    if (value_seq.size() != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Get value sequence size " << value_seq.size() << " not equal to abstract size "
                        << abs_seq->size();
    }
    for (size_t i = 0; i < value_seq.size(); ++i) {
      auto sub_value = value_seq[i];
      CNodePtr new_din = ad_param_->tape_->FuncGraph::NewCNode(
        {NewValueNode(prim::kPrimTupleGetItem), din, NewValueNode(SizeToLong(i))});
      new_din->set_abstract(CommonUtils::SetAbstractValueToAnyValue(abs_seq->elements()[i]));
      if (din == fn->fake_dout()) {
        // The new_din's index input is fn->fake_dout()
        LazyAddUser(fn->fake_dout(), new_din, 1);
      }
      // Add next edge to fn
      UpdateNextEdge(fn, new_din, sub_value, abs_seq->elements()[i]);
    }
  } else if (input_arg->isa<tensor::COOTensor>()) {
    auto input_tensor = input_arg->cast<tensor::COOTensorPtr>()->GetIndices();
    UpdateNextEdge(fn, din, input_tensor, CommonUtils::SetAbstractValueToAnyValue(input_tensor->ToAbstract()));
  } else if (input_arg->isa<tensor::CSRTensor>()) {
    auto input_tensor = input_arg->cast<tensor::CSRTensorPtr>()->GetIndices();
    UpdateNextEdge(fn, din, input_tensor, CommonUtils::SetAbstractValueToAnyValue(input_tensor->ToAbstract()));
  } else if (input_arg->isa<ValueDictionary>()) {
    UpdateNextEdgeForDict(fn, din, input_arg, abs);
  } else {
    MS_LOG(DEBUG) << "It is not tensor, not need derivation " << input_arg->ToString();
    return;
  }
}

AnfNodePtr IrBprop::TraceInput(const IrFunctionNodePtr &fn, const ValuePtr &out_value,
                               const abstract::AbstractBasePtr &out_abs, const tensor::BaseTensorPtr &input_tensor,
                               const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(out_value);
  MS_EXCEPTION_IF_NULL(out_abs);
  MS_EXCEPTION_IF_NULL(input_tensor);
  MS_EXCEPTION_IF_NULL(din);

  // The node corresponding output tensor is the same as the currently used tensor
  if (out_value->isa<tensor::BaseTensor>()) {
    // out_value is be used, may be it is one of multiple output
    auto out_tensor = out_value->cast<tensor::BaseTensorPtr>();
    if (input_tensor->id() == out_tensor->id()) {
      return din;
    }
    return PyNativeAlgo::AutoGradUtil::BuildSpecialNode(ad_param_->tape_, out_value, out_abs,
                                                        SpecialType::kZerosLikeType);
  }
  if (out_value->isa<ValueSequence>()) {
    // The corresponding output of node is ValueSequence, but used one of it
    AnfNodePtrList inputs;
    (void)inputs.emplace_back(NewValueNode(prim::kPrimMakeTuple));
    auto value_seq = out_value->cast<ValueSequencePtr>();
    auto abs_seq = out_abs->cast<abstract::AbstractSequencePtr>();
    if (abs_seq == nullptr) {
      MS_LOG(EXCEPTION) << "Get output abstract " << out_abs->ToString() << ", not abstract sequence";
    }
    int index = -1;
    for (size_t i = 0; i < value_seq->size(); ++i) {
      // Find the value's din, if value equal to sub_value, means value be used, is it will get din; Otherwise value's
      // din is zero , which set by second branch condition above
      auto new_din = TraceInput(fn, value_seq->value()[i], abs_seq->elements()[i], input_tensor, din);
      (void)inputs.emplace_back(new_din);

      // if exist din == fake_dout, we record it in user vector
      if (din == fn->fake_dout() && new_din == din) {
        index = static_cast<int>(inputs.size()) - 1;
      }
    }
    auto new_din = ad_param_->tape_->FuncGraph::NewCNode(inputs);
    new_din->set_abstract(out_abs);
    if (index != -1) {
      LazyAddUser(fn->fake_dout(), new_din, index);
    }
    return new_din;
  }
  if (out_value->isa<ValueDictionary>()) {
    return TraceInputForDict(fn, out_value, out_abs, input_tensor, din);
  }
  MS_LOG(DEBUG) << "Get non tensor input " << out_value->ToString();
  return PyNativeAlgo::AutoGradUtil::BuildSpecialNode(ad_param_->tape_, out_value, out_abs,
                                                      SpecialType::kZerosLikeType);
}

AnfNodePtr IrBprop::TraceInputForDict(const IrFunctionNodePtr &fn, const ValuePtr &out_value,
                                      const abstract::AbstractBasePtr &out_abs,
                                      const tensor::BaseTensorPtr &input_tensor, const AnfNodePtr &din) {
  // The corresponding output of node is ValueDictionary, but used one of it
  AnfNodePtrList key_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  AnfNodePtrList value_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  abstract::AbstractBasePtrList local_key_abs_inputs;
  abstract::AbstractBasePtrList local_value_abs_inputs;
  auto value_dict = out_value->cast<ValueDictionaryPtr>();
  auto abs_dict = out_abs->cast<abstract::AbstractDictionaryPtr>();
  MS_EXCEPTION_IF_NULL(abs_dict);
  int index = -1;
  for (size_t i = 0; i < value_dict->size(); ++i) {
    // Find the value's din, if value equal to sub_value, means value be used, is it will get din; Otherwise value's
    // din is zero, which set by second branch condition above
    (void)key_inputs.emplace_back(
      PyNativeAlgo::Common::CreateValueNodeByValue(value_dict->value()[i].first, abs_dict->elements()[i].first));
    (void)local_key_abs_inputs.emplace_back(abs_dict->elements()[i].first);
    auto new_din = TraceInput(fn, value_dict->value()[i].second, abs_dict->elements()[i].second, input_tensor, din);
    (void)value_inputs.emplace_back(new_din);
    (void)local_value_abs_inputs.emplace_back(abs_dict->elements()[i].second);

    // if exist din == fake_dout, we record it in user vector
    if (din == fn->fake_dout() && new_din == din) {
      index = static_cast<int>(value_inputs.size()) - 1;
    }
  }
  auto local_key_node = ad_param_->tape_->NewCNode(key_inputs);
  local_key_node->set_abstract(std::make_shared<abstract::AbstractTuple>(local_key_abs_inputs));
  auto local_value_node = ad_param_->tape_->NewCNode(value_inputs);
  local_value_node->set_abstract(std::make_shared<abstract::AbstractTuple>(local_value_abs_inputs));
  auto new_din = ad_param_->tape_->NewCNode({NewValueNode(prim::kPrimMakeDict), local_key_node, local_value_node});
  new_din->set_abstract(abs_dict);
  if (index != -1) {
    LazyAddUser(fn->fake_dout(), local_value_node, index);
  }
  return new_din;
}

void IrBprop::AddTupleGetItemUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  (void)ad_param_->users_.tuple_getitem_user_[node].emplace_back(user, index);
}

void IrBprop::UpdateLazyUser() {
  // For lazy add user data, we need emplace to user.
  for (const auto &user_data : ad_param_->lazy_user_data_) {
    AddUser(std::get<kIndex0>(user_data), std::get<kIndex1>(user_data), std::get<kIndex2>(user_data));
  }
}

void IrBprop::LazyAddUser(const AnfNodePtr &node, const CNodePtr &user, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(user);
  (void)ad_param_->lazy_user_data_.emplace_back(node, user, index);
}
}  // namespace mindspore::pynative::autograd
