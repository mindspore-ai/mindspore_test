/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_UTILS_H_
#define MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_UTILS_H_

#include "pipeline/pynative/grad/grad_utils.h"
#include <algorithm>
#include <vector>
#include "mindspore/ops/op_def/sparse_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/array_op_name.h"
#include "pybind_api/ir/primitive_py.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "pipeline/pynative/grad/hook_py.h"
#include "utils/ms_context.h"
#include "include/common/utils/utils.h"
#include "pipeline/jit/ps/pipeline.h"
#include "include/common/utils/convert_utils_py.h"
#include "frontend/optimizer/fallback_rewriter.h"
#include "pipeline/pynative/grad/jit/jit_grad.h"
#include "frontend/optimizer/environ_conversion.h"
#include "mindspore/ops/op_def/sequence_op_name.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "include/common/pynative/abstract_converter.h"
#include "mindspore/ccsrc/pyboost/auto_generate/clone.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "pipeline/pynative/pynative_utils.h"
#include "mindspore/ccsrc/pyboost/functions/auto_generate/functions.h"
#include "mindspore/ccsrc/pyboost/functions/auto_grad_guard.h"

namespace mindspore {
namespace pynative {
namespace PyNativeAlgo {
using CallBackFn = std::function<VectorRef(const VectorRef &arg_list)>;
const mindspore::HashSet<std::string> kGradBlackList{kMakeTupleOpName,         kMakeListOpName,
                                                     kTupleGetItemOpName,      kStopGradientOpName,
                                                     kUpdateStateOpName,       kNPUAllocFloatStatusOpName,
                                                     kNPUGetFloatStatusOpName, kNPUClearFloatStatusOpName,
                                                     kZerosLikeExtOpName,      kOnesLikeExtOpName};
mindspore::HashMap<std::string, pipeline::ResourcePtr> jit_call_graph_compile_cache_;

// for simply infer (simple infer will push abs in bprop queue)
static AbstractConverter kGradAbstractConverter;
using AutoGradMetaData = autograd::AutoGradMetaData;
using ViewAutoGradMetaData = autograd::ViewAutoGradMetaData;
using ViewAutoGradMetaDataPtr = std::shared_ptr<ViewAutoGradMetaData>;
using ViewInfo = autograd::ViewInfo;

class CloneFuncRegister {
 public:
  CloneFuncRegister() { kernel::pyboost::RegisterCloneFunc(AutoGradUtil::CheckAndCloneInplaceInput); }
};
static CloneFuncRegister clone_func_register;
namespace {
AnfNodePtr CreateMakeTupleNode(const KernelGraphPtr &tape, const ValueSequencePtr &tuple,
                               const abstract::AbstractSequencePtr &abs_seq, const SpecialType &type) {
  AnfNodePtrList args{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < tuple->size(); ++i) {
    AnfNodePtr special_like_value =
      AutoGradUtil::BuildSpecialNode(tape, tuple->value()[i], abs_seq->elements()[i], type);
    (void)args.emplace_back(special_like_value);
  }
  auto special_like_value = tape->FuncGraph::NewCNode(args);
  special_like_value->set_abstract(abs_seq);
  return special_like_value;
}

AnfNodePtr CreateMakeDictNode(const KernelGraphPtr &tape, const ValueDictionaryPtr &v_dict,
                              const abstract::AbstractDictionaryPtr &abs_dict, const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(tape);
  MS_EXCEPTION_IF_NULL(v_dict);
  MS_EXCEPTION_IF_NULL(abs_dict);
  AnfNodePtrList key_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  AnfNodePtrList value_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  abstract::AbstractBasePtrList local_key_abs_inputs;
  abstract::AbstractBasePtrList local_value_abs_inputs;
  for (size_t i = 0; i < v_dict->size(); ++i) {
    (void)key_inputs.emplace_back(
      Common::CreateValueNodeByValue(v_dict->value()[i].first, abs_dict->elements()[i].first));
    (void)local_key_abs_inputs.emplace_back(abs_dict->elements()[i].first);
    AnfNodePtr special_like_value =
      AutoGradUtil::BuildSpecialNode(tape, v_dict->value()[i].second, abs_dict->elements()[i].second, type);
    (void)value_inputs.emplace_back(special_like_value);
    (void)local_value_abs_inputs.emplace_back(abs_dict->elements()[i].second);
  }
  auto local_key_node = tape->NewCNode(key_inputs);
  local_key_node->set_abstract(std::make_shared<abstract::AbstractTuple>(local_key_abs_inputs));
  auto local_value_node = tape->NewCNode(value_inputs);
  local_value_node->set_abstract(std::make_shared<abstract::AbstractTuple>(local_value_abs_inputs));
  auto dict_node = tape->NewCNode({NewValueNode(prim::kPrimMakeDict), local_key_node, local_value_node});
  dict_node->set_abstract(abs_dict);
  return dict_node;
}

ValuePtr WrapCOOTensor(const ValuePtr &coo_out, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(coo_out);
  auto coo_tensor = coo_out->cast<tensor::COOTensorPtr>();
  MS_EXCEPTION_IF_NULL(coo_tensor);
  auto value_tensor = value->cast<tensor::TensorPtr>();
  if (value_tensor == nullptr) {
    auto base_tensor = value->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(base_tensor);
    value_tensor = std::make_shared<tensor::Tensor>(*base_tensor);
  }
  auto indices_tensor = coo_tensor->GetIndices();
  auto shape_vector = coo_tensor->shape();
  return std::make_shared<tensor::COOTensor>(indices_tensor, value_tensor, shape_vector);
}

ValuePtr WrapCSRTensor(const ValuePtr &csr_out, const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(csr_out);
  auto csr_tensor = csr_out->cast<tensor::CSRTensorPtr>();
  MS_EXCEPTION_IF_NULL(csr_tensor);
  auto value_tensor = value->cast<tensor::TensorPtr>();
  if (value_tensor == nullptr) {
    auto base_tensor = value->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(base_tensor);
    value_tensor = std::make_shared<tensor::Tensor>(*base_tensor);
  }
  auto indptr_tensor = csr_tensor->GetIndptr();
  auto indices_tensor = csr_tensor->GetIndices();
  auto shape_vector = csr_tensor->shape();
  return std::make_shared<tensor::CSRTensor>(indptr_tensor, indices_tensor, value_tensor, shape_vector);
}

ValueNodePtr GetSparseTensorShapeNode(const ShapeVector &shape) {
  auto value_shape = NewValueNode(shape);
  std::vector<abstract::AbstractBasePtr> abstract_shape;
  (void)std::transform(
    shape.begin(), shape.end(), std::back_inserter(abstract_shape),
    [](auto shp) -> abstract::AbstractScalarPtr { return std::make_shared<abstract::AbstractScalar>(shp); });
  auto abs_shape = std::make_shared<abstract::AbstractTuple>(abstract_shape);
  value_shape->set_abstract(abs_shape);
  return value_shape;
}

void ConvertSimpleInferInfoToAbstract(const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(op_grad_info);
  // Get inputs abstract
  for (const auto &v : op_grad_info->input_value) {
    op_grad_info->input_abs.emplace_back(kGradAbstractConverter.ConvertAbstract(v));
  }

  // Get output abstract
  MS_EXCEPTION_IF_NULL(op_grad_info->output_value_simple_info);
  op_grad_info->out_abs = TransformValueSimpleInfoToAbstract(*op_grad_info->output_value_simple_info);

  // Set abstract to tensor
  AutoGradUtil::CacheOutputAbstract(op_grad_info->out_value, op_grad_info->out_abs);
  MS_LOG(DEBUG) << "Get output abstract " << op_grad_info->out_abs->ToString();
}
}  // namespace

InputType AutoGradUtil::SetValueGradInfo(const ValuePtr &value, InputType grad_type) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    const auto &tensor_value = value->cast<tensor::BaseTensorPtr>();
    auto auto_grad_meta_data = autograd::impl::get_autograd_meta_impl(tensor_value);
    if (auto_grad_meta_data != nullptr) {
      if (auto_grad_meta_data->input_type() != InputType::kUnkown) {
        return auto_grad_meta_data->input_type();
      }
      MS_LOG(DEBUG) << "Set input type for tensor " << tensor_value->id();
    } else if (grad_type != InputType::kConstant || tensor_value->is_parameter()) {
      MS_LOG(DEBUG) << "Create new auto grad meta for tensor " << tensor_value->id();
      auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
      autograd::RegisterHook::UpdateTensorBackwardHook(auto_grad_meta_data, tensor_value->id());
      tensor_value->set_auto_grad_meta_data(auto_grad_meta_data);
    }
    // Scalar tensor auto grad meta data is nullptr
    if (auto_grad_meta_data != nullptr) {
      if (tensor_value->is_parameter() && grad_type != InputType::kInput) {
        grad_type = InputType::kParameter;
      }
      auto_grad_meta_data->set_input_type(grad_type);
    }
    return grad_type;
  }
  if (value->isa<ValueSequence>()) {
    const auto &value_seq = value->cast<ValueSequencePtr>()->value();
    InputType ret_type = grad_type;
    for (const auto &v : value_seq) {
      auto ret = SetValueGradInfo(v, grad_type);
      if (IsParam(ret)) {
        ret_type = ret;
      }
    }
    return ret_type;
  }
  if (value->isa<tensor::COOTensor>()) {
    const auto &coo_tensor = value->cast<tensor::COOTensorPtr>();
    const auto &indices_tensor = coo_tensor->GetIndices();
    return SetValueGradInfo(indices_tensor, grad_type);
  }
  if (value->isa<tensor::CSRTensor>()) {
    const auto &csr_tensor = value->cast<tensor::CSRTensorPtr>();
    const auto &indices_tensor = csr_tensor->GetIndices();
    return SetValueGradInfo(indices_tensor, grad_type);
  }
  if (value->isa<ValueDictionary>()) {
    const auto &dic_v = value->cast<ValueDictionaryPtr>()->value();
    for (const auto &v : dic_v) {
      (void)SetValueGradInfo(v.second, grad_type);
    }
  }
  return grad_type;
}

InputType AutoGradUtil::SetTensorGradInfo(const tensor::BaseTensorPtr &tensor) {
  MS_EXCEPTION_IF_NULL(tensor);
  auto auto_grad_meta_data = autograd::impl::get_autograd_meta_impl(tensor);
  if (auto_grad_meta_data != nullptr) {
    if (auto_grad_meta_data->input_type() != InputType::kUnkown) {
      return auto_grad_meta_data->input_type();
    }
    MS_LOG(DEBUG) << "Set input type for tensor " << tensor->id();
  } else if (tensor->is_parameter()) {
    MS_LOG(DEBUG) << "Create new auto grad meta for tensor " << tensor->id();
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    autograd::RegisterHook::UpdateTensorBackwardHook(auto_grad_meta_data, tensor->id());
    tensor->set_auto_grad_meta_data(auto_grad_meta_data);
  }
  // Set weight tensor grad type
  if (tensor->is_parameter()) {
    auto_grad_meta_data->set_input_type(InputType::kParameter);
    return InputType::kParameter;
  }
  return InputType::kConstant;
}

bool AutoGradUtil::IsPrimNeedGrad(const PrimitivePtr &prim) {
  MS_EXCEPTION_IF_NULL(prim);
  return kGradBlackList.find(prim->name()) == kGradBlackList.end();
}

ValuePtr AutoGradUtil::BaseRefToValue(const BaseRef &value, bool requires_grad, bool is_out_sequence, size_t op_index) {
  MS_EXCEPTION_IF_NULL(value);
  ValuePtr ret;
  if (utils::isa<tensor::BaseTensorPtr>(value)) {
    auto t = utils::cast<tensor::BaseTensorPtr>(value);
    if (requires_grad) {
      t->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>(op_index, InputType::kOpOutput));
    }
    ret = t;
  } else if (utils::isa<ValuePtr>(value)) {
    ret = utils::cast<ValuePtr>(value);
  } else if (utils::isa<VectorRef>(value)) {
    auto vec_ref = utils::cast<VectorRef>(value);
    ret = VectorRefToValue(vec_ref, requires_grad, is_out_sequence, op_index);
  } else if (utils::isa<int>(value)) {
    ret = MakeValue(utils::cast<int>(value));
  } else if (utils::isa<float>(value)) {
    ret = MakeValue(utils::cast<float>(value));
  } else if (utils::isa<double>(value)) {
    ret = MakeValue(utils::cast<double>(value));
  } else if (utils::isa<bool>(value)) {
    ret = MakeValue(utils::cast<bool>(value));
  } else {
    MS_LOG(EXCEPTION) << "value is not support type " << value.ToString();
  }
  return ret;
}

ValuePtr AutoGradUtil::VectorRefToValue(const VectorRef &vec_ref, bool requires_grad, bool is_out_sequence,
                                        size_t op_index) {
  MS_EXCEPTION_IF_NULL(vec_ref);
  size_t value_size = vec_ref.size();
  if (value_size == 1 && !is_out_sequence) {
    return BaseRefToValue(vec_ref[0], requires_grad, is_out_sequence, op_index);
  }
  std::vector<ValuePtr> v_list(value_size);
  for (size_t i = 0; i < value_size; ++i) {
    v_list[i] = BaseRefToValue(vec_ref[i], requires_grad, is_out_sequence, op_index);
  }
  return std::make_shared<ValueTuple>(v_list);
}

void AutoGradUtil::BuildViewAutoGradMeta(const tensor::BaseTensorPtr &src_tensor, const tensor::BaseTensorPtr &output,
                                         size_t op_index, autograd::CreationType creation_type) {
  MS_EXCEPTION_IF_NULL(output);
  auto view_meta = autograd::impl::get_view_autograd_meta_impl(src_tensor);
  if (view_meta != nullptr) {
    output->set_version(src_tensor->version());
    output->set_auto_grad_meta_data(std::make_shared<autograd::ViewAutoGradMetaData>(
      view_meta->view_info().Union(), op_index, InputType::kOpOutput,
      creation_type != autograd::CreationType::kDefault ? creation_type : view_meta->creation_type()));
  } else {
    if (src_tensor->auto_grad_meta_data() == nullptr) {
      // If base tensor is input of view op, we need construct auto_grad_meta_data for base tensor, to
      // avoid view tensor being inplaced by inpalce op, which will need update grad info of base tensor.
      // we need construct auto_grad_meta_data in second thread rather than bprop thread.
      MS_LOG(DEBUG) << "Create new auto grad meta for input tensor of view op " << src_tensor->id();
      auto auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
      src_tensor->set_auto_grad_meta_data(auto_grad_meta_data);
      if (src_tensor->is_parameter()) {
        autograd::RegisterHook::UpdateTensorBackwardHook(auto_grad_meta_data, src_tensor->id());
      }
    }
    // Temp method to avoid view tensor hold by grad.
    auto base_tensor = std::make_shared<tensor::BaseTensor>(*src_tensor);
    if (src_tensor->is_parameter()) {
      base_tensor->set_param_info(src_tensor->param_info());
    }
    base_tensor->set_device_address(nullptr);
    ViewInfo view_info(base_tensor);
    output->set_version(src_tensor->version());
    output->set_auto_grad_meta_data(std::make_shared<autograd::ViewAutoGradMetaData>(
      std::move(view_info), op_index, InputType::kOpOutput, creation_type));
  }
}

void AutoGradUtil::SetInferOutputToGrad(const OpGradInfoPtr &op_grad_info, const kernel::pyboost::OpPtr &op) {
  if (op->output_value_simple_info() != nullptr) {
    op_grad_info->output_value_simple_info = op->output_value_simple_info();
    op_grad_info->output_value_simple_info->is_tuple_output_ = false;
  }
}

void AutoGradUtil::SetInferMultiOutputToGrad(const OpGradInfoPtr &op_grad_info, const kernel::pyboost::OpPtr &op) {
  if (op->output_value_simple_info() != nullptr) {
    op_grad_info->output_value_simple_info = op->output_value_simple_info();
    op_grad_info->output_value_simple_info->is_tuple_output_ = true;
  }
}

ValuePtr AutoGradUtil::MakeOutput(bool requires_grad, const kernel::pyboost::OpPtr &op, size_t op_index,
                                  const tensor::BaseTensorPtr &base_view) {
  // delete NoneTypeNode check.
  if (base_view != nullptr && op->outputs()[0]->storage_info() != nullptr) {
    autograd::CreationType creationType =
      requires_grad ? autograd::CreationType::kDefault : autograd::CreationType::kNoGradMode;
    BuildViewAutoGradMeta(base_view, op->outputs()[0], op_index, creationType);
  } else if (requires_grad) {
    if (op->outputs()[0]->auto_grad_meta_data() == nullptr) {
      op->outputs()[0]->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>(op_index, InputType::kOpOutput));
    }
  }
  return op->outputs()[0];
}

ValuePtr AutoGradUtil::MakeMultiOutput(bool requires_grad, const kernel::pyboost::OpPtr &op, size_t op_index,
                                       const tensor::BaseTensorPtr &base_view) {
  size_t size = op->outputs().size();
  std::vector<ValuePtr> output_values(size);
  for (size_t i = 0; i < size; ++i) {
    const auto &output_tensor = op->outputs()[i];
    MS_EXCEPTION_IF_NULL(output_tensor);
    // Set auto grad meta data for op outputs
    if (base_view != nullptr && output_tensor->storage_info() != nullptr) {
      BuildViewAutoGradMeta(base_view, output_tensor, op_index, autograd::CreationType::kMultiOutput);
    } else if (requires_grad) {
      if (op->outputs()[i]->auto_grad_meta_data() == nullptr) {
        op->outputs()[i]->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>(op_index, InputType::kOpOutput));
      }
    }
    output_values[i] = output_tensor;
  }
  return std::make_shared<ValueTuple>(output_values);
}

ValuePtr AutoGradUtil::MakeMultiOutput(bool requires_grad, const kernel::pyboost::OpPtr &op, size_t op_index,
                                       const ValueTuplePtr &base_view) {
  size_t size = op->outputs().size();
  std::vector<ValuePtr> output_values(size);
  auto inputs = base_view->value();
  if (inputs.size() != size) {
    MS_LOG(EXCEPTION) << "For multi inputs and multi outputs view op, inputs size should be same as outputs!";
  }
  for (size_t i = 0; i < size; ++i) {
    const auto &output_tensor = op->outputs()[i];
    MS_EXCEPTION_IF_NULL(output_tensor);
    const auto input_tensor = inputs[i]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(input_tensor);
    // Set auto grad meta data for op output
    if (input_tensor != nullptr && output_tensor->storage_info() != nullptr) {
      BuildViewAutoGradMeta(input_tensor, output_tensor, op_index, autograd::CreationType::kDefault);
    } else if (requires_grad) {
      if (op->outputs()[i]->auto_grad_meta_data() == nullptr) {
        op->outputs()[i]->set_auto_grad_meta_data(std::make_shared<AutoGradMetaData>(op_index, InputType::kOpOutput));
      }
    }
    output_values[i] = output_tensor;
  }
  return std::make_shared<ValueTuple>(output_values);
}

void AutoGradUtil::BumpVersion(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  auto tensor = value->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  tensor->BumpVersion();
}

bool AutoGradUtil::NeedGrad(const tensor::BaseTensorPtr &input_tensor) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  if (IsParamRequiresGrad(input_tensor)) {
    return true;
  }
  auto auto_grad_meta_data = input_tensor->auto_grad_meta_data();
  if (auto_grad_meta_data != nullptr) {
    auto variable = auto_grad_meta_data->UnsafeGetVariableImpl();
    if (variable != nullptr) {
      return true;
    }
  }
  return false;
}

bool AutoGradUtil::NeedGrad(const std::vector<ValuePtr> &input_values) {
  for (const ValuePtr &input_arg : input_values) {
    MS_EXCEPTION_IF_NULL(input_arg);
    if (input_arg->isa<tensor::BaseTensor>()) {
      const auto input_tensor = input_arg->cast<tensor::BaseTensorPtr>();
      if (NeedGrad(input_tensor)) {
        return true;
      }
    } else if (input_arg->isa<ValueSequence>()) {
      auto value_seq = input_arg->cast<ValueSequencePtr>()->value();
      if (NeedGrad(value_seq)) {
        return true;
      }
    } else if (input_arg->isa<tensor::COOTensor>() || input_arg->isa<tensor::CSRTensor>()) {
      return true;
    } else if (input_arg->isa<ValueDictionary>()) {
      auto dict_val = input_arg->cast<ValueDictionaryPtr>()->value();
      for (auto kv : dict_val) {
        if (NeedGrad({kv.second})) {
          return true;
        }
      }
    }
    MS_LOG(DEBUG) << "Get value " << input_arg->ToString();
  }
  return false;
}

bool AutoGradUtil::IsZerosLikeNode(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!node->isa<CNode>()) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  if (IsPrimitiveCNode(cnode, prim::kPrimZerosLike)) {
    return true;
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || IsPrimitiveCNode(cnode, prim::kPrimMakeList)) {
    return std::all_of(cnode->inputs().begin() + 1, cnode->inputs().end(),
                       [](const auto &node) { return IsZerosLikeNode(node) == true; });
  }
  if (IsPrimitiveCNode(cnode, prim::kPrimMakeDict)) {
    return IsZerosLikeNode(cnode->input(kIndex2));
  }
  return false;
}

ValuePtr AutoGradUtil::GetFakeZeroTensor() {
  static ValuePtr fake_v = std::make_shared<tensor::Tensor>(0);
  return fake_v;
}

ValuePtr AutoGradUtil::BuildSpecialValueGrad(const ValuePtr &value, const tensor::BaseTensorPtr &grad,
                                             autograd::FuncBuilder *func_builder, const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(value);
  if (grad != nullptr) {
    return grad;
  }
  if (value->isa<tensor::BaseTensor>()) {
    const auto tensor = value->cast<tensor::BaseTensorPtr>();
    return (type == SpecialType::kZerosLikeType ? func_builder->Zeros(tensor) : func_builder->Ones(tensor));
  }
  if (value->isa<ValueSequence>()) {
    ValuePtr zero_value = nullptr;
    auto v_seq = value->cast<ValueSequencePtr>();
    ValuePtrList v_list;
    for (const auto &item : v_seq->value()) {
      (void)v_list.emplace_back(BuildSpecialValueGrad(item, grad, func_builder, type));
    }
    return std::make_shared<ValueTuple>(v_list);
  }
  if (value->isa<Scalar>()) {
    auto fake_tensor = std::make_shared<tensor::Tensor>(0, value->type());
    return BuildSpecialValueGrad(fake_tensor, grad, func_builder, type);
  }
  if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    return WrapCSRTensor(csr_tensor, BuildSpecialValueGrad(csr_tensor->GetValues(), grad, func_builder, type));
  }
  if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    return WrapCOOTensor(coo_tensor, BuildSpecialValueGrad(coo_tensor->GetValues(), grad, func_builder, type));
  }
  MS_LOG(INFO) << "For value " << value->ToString() << ", the type is not tensor or scalar";
  auto fake_tensor = std::make_shared<tensor::Tensor>(0, value->type());
  return BuildSpecialValueGrad(fake_tensor, grad, func_builder, type);
}

AnfNodePtr AutoGradUtil::BuildSpecialNode(const KernelGraphPtr &tape, const ValuePtr &value,
                                          const abstract::AbstractBasePtr &abs, const SpecialType &type) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::BaseTensor>()) {
    auto prim_node =
      (type == SpecialType::kZerosLikeType ? NewValueNode(std::make_shared<Primitive>(*prim::kPrimZerosLike))
                                           : NewValueNode(std::make_shared<Primitive>(*prim::kPrimOnesLike)));
    auto value_node = Common::CreateValueNodeByValue(value, abs);
    auto special_like_value = tape->FuncGraph::NewCNode({prim_node, value_node});
    special_like_value->set_abstract(value_node->abstract());
    return special_like_value;
  }
  if (value->isa<ValueSequence>()) {
    auto tuple = value->cast<ValueSequencePtr>();
    abstract::AbstractSequencePtr abs_seq;
    if (abs == nullptr) {
      abs_seq = Common::SetAbstractValueToAnyValue(value->ToAbstract())->cast<abstract::AbstractSequencePtr>();
    } else {
      abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    }
    return CreateMakeTupleNode(tape, tuple, abs_seq, type);
  }
  if (value->isa<Scalar>()) {
    auto fake_tensor = GetFakeZeroTensor();
    return BuildSpecialNode(tape, fake_tensor, nullptr, type);
  }
  if (value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = value->cast<tensor::CSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_tensor);
    auto data = csr_tensor->GetValues();
    return BuildSpecialNode(tape, data, nullptr, type);
  }
  if (value->isa<tensor::COOTensor>()) {
    auto coo_tensor = value->cast<tensor::COOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_tensor);
    auto data = coo_tensor->GetValues();
    return BuildSpecialNode(tape, data, nullptr, type);
  }
  if (value->isa<ValueDictionary>()) {
    auto v_dict = value->cast<ValueDictionaryPtr>();
    abstract::AbstractDictionaryPtr abs_dict;
    if (abs == nullptr) {
      abs_dict = Common::SetAbstractValueToAnyValue(value->ToAbstract())->cast<abstract::AbstractDictionaryPtr>();
    } else {
      abs_dict = abs->cast<abstract::AbstractDictionaryPtr>();
    }
    return CreateMakeDictNode(tape, v_dict, abs_dict, type);
  }
  MS_LOG(INFO) << "For value " << value->ToString() << ", the type is not tensor or scalar";
  return BuildSpecialNode(tape, GetFakeZeroTensor(), nullptr, type);
}

AnfNodePtr AutoGradUtil::BuildSparseTensorNode(const KernelGraphPtr &tape, const ValuePtr &sparse_value,
                                               const AnfNodePtr &dout_value_node) {
  MS_EXCEPTION_IF_NULL(tape);
  MS_EXCEPTION_IF_NULL(sparse_value);
  if (sparse_value->isa<tensor::CSRTensor>()) {
    auto csr_tensor = sparse_value->cast<tensor::CSRTensorPtr>();
    MS_EXCEPTION_IF_NULL(csr_tensor);
    auto indptr_node = Common::CreateValueNodeByValue(csr_tensor->GetIndptr());
    auto indices_node = Common::CreateValueNodeByValue(csr_tensor->GetIndices());
    auto value_shape = GetSparseTensorShapeNode(csr_tensor->shape());
    auto special_like_csr_node = tape->FuncGraph::NewCNode(
      {NewValueNode(prim::kPrimMakeTuple), indptr_node, indices_node, dout_value_node, value_shape});
    special_like_csr_node->set_abstract(sparse_value->ToAbstract()->Broaden());
    return special_like_csr_node;
  }
  if (sparse_value->isa<tensor::COOTensor>()) {
    auto coo_tensor = sparse_value->cast<tensor::COOTensorPtr>();
    MS_EXCEPTION_IF_NULL(coo_tensor);
    auto indices_node = Common::CreateValueNodeByValue(coo_tensor->GetIndices());
    auto value_shape = GetSparseTensorShapeNode(coo_tensor->shape());
    auto special_like_coo_node =
      tape->FuncGraph::NewCNode({NewValueNode(prim::kPrimMakeTuple), indices_node, dout_value_node, value_shape});
    special_like_coo_node->set_abstract(sparse_value->ToAbstract()->Broaden());
    return special_like_coo_node;
  }
  MS_LOG(EXCEPTION) << "Get invalid sparse tensor";
}

void AutoGradUtil::SetGradMetaData(const ValuePtr &value, const VariablePtr &variable, const ParameterPtr &param) {
  if (value->isa<tensor::BaseTensor>()) {
    tensor::BaseTensorPtr tensor = nullptr;
    tensor = value->cast<tensor::BaseTensorPtr>();
    auto auto_grad_meta_data = tensor->auto_grad_meta_data();
    if (auto_grad_meta_data == nullptr) {
      MS_LOG(DEBUG) << "tensor has no auto_grad_meta_data";
      auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
      tensor->set_auto_grad_meta_data(auto_grad_meta_data);
    }
    auto_grad_meta_data->set_variable(variable);
    if (param != nullptr) {
      auto_grad_meta_data->set_parameter(param);
      auto_grad_meta_data->set_input_type(InputType::kParameter);
    }
  } else if (value->isa<ValueSequence>()) {
    auto value_sequence = value->cast<ValueSequencePtr>();
    for (const auto &val : value_sequence->value()) {
      SetGradMetaData(val, variable);
    }
  } else if (value->isa<ValueDictionary>()) {
    auto value_dict = value->cast<ValueDictionaryPtr>();
    for (const auto &val : value_dict->value()) {
      SetGradMetaData(val.second, variable);
    }
  }
}

void AutoGradUtil::SetGradInfoForInputs(const ValuePtr &value, const VariablePtr &variable,
                                        autograd::MetaGradInfoList *param_meta_grad_info, const ParameterPtr &param) {
  if (value->isa<tensor::BaseTensor>()) {
    const auto &input_tensor = value->cast<tensor::BaseTensorPtr>();
    const auto &auto_grad_meta_data = autograd::impl::get_autograd_meta_impl(input_tensor);
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto_grad_meta_data->set_variable(variable);
    auto_grad_meta_data->set_parameter(param);
    (*param_meta_grad_info)[input_tensor] = auto_grad_meta_data;
  } else if (value->isa<tensor::COOTensor>()) {
    const auto &coo_tensor = value->cast<tensor::COOTensorPtr>();
    const auto &indices_tensor = coo_tensor->GetIndices();
    SetGradInfoForInputs(indices_tensor, variable, param_meta_grad_info, param);
  } else if (value->isa<tensor::CSRTensor>()) {
    const auto &csr_tensor = value->cast<tensor::CSRTensorPtr>();
    const auto &indices_tensor = csr_tensor->GetIndices();
    SetGradInfoForInputs(indices_tensor, variable, param_meta_grad_info, param);
  }
}

// Create fake bprop
void AutoGradUtil::BuildFakeBpropCNode(const CNodePtr &cnode, std::vector<CNodePtr> *outputs) {
  auto prim = GetCNodePrimitive(cnode);
  if (prim == nullptr) {
    MS_LOG(EXCEPTION) << "Should be primitive, but: " << cnode->DebugString();
  }
  size_t dout_index = cnode->size() - 1;
  const auto &dout = cnode->input(dout_index);
  const auto &dout_cnode = dout->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dout_cnode);
  // Size is same as op_arg size
  size_t input_size = cnode->size() - 2;
  for (size_t i = 1; i < input_size; ++i) {
    (void)outputs->emplace_back(dout_cnode);
  }
}

CallBackFn AutoGradUtil::CreateGraphCallBack(const FuncGraphPtr &call_graph, const std::string &cache_key,
                                             const GraphCallCondition &graph_call_condition) {
  // kFlagJitCallGraph is set true to avoid compilig call_graph whe compiling the main graph
  call_graph->set_flag(kFlagJitCallGraph, true);
  // call graph not inline to grad top
  call_graph->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  // Pynative bprop graph flag
  call_graph->set_flag(kFlagIsPynativeBpropGraph, true);
  // Run graph by single op will use this kFlagPyNativeBpropGraphWithBpropCut flag
  if (graph_call_condition.is_dynamic_shape_process_) {
    call_graph->set_flag(kFlagPyNativeBpropGraphWithBpropCut, false);
    if (!graph_call_condition.is_jit_graph_) {
      call_graph->set_flag(kFlagEnableRunGraphBySingleOp, true);
    }
  }
  pipeline::ResourcePtr resource;
  constexpr auto kNeedCompile = "NeedCompile";
  const auto it = jit_call_graph_compile_cache_.find(cache_key);
  bool need_compile = (it == jit_call_graph_compile_cache_.end());
  if (need_compile) {
    resource = std::make_shared<pipeline::Resource>();
    resource->set_func_graph(call_graph);
    if (graph_call_condition.is_func_grad_) {
      auto manager = resource->manager();
      manager->AddFuncGraph(call_graph, false);
      (void)opt::EnvironConversion(resource);
      if (graph_call_condition.jit_out_has_dict_) {
        MS_LOG(DEBUG) << "Jit out is dict, need convert make dict to pyexecute";
        (void)mindspore::opt::RewriterAfterOptA(resource->func_graph(), resource);
      }
    }
    if (graph_call_condition.is_jit_graph_ || !graph_call_condition.is_dynamic_shape_process_) {
      (void)jit_call_graph_compile_cache_.emplace(cache_key, resource);
    }
    resource->SetResult(kNeedCompile, true);
  } else {
    resource = it->second;
    // If resource func graph not compile(not call run grad graph), but hit cache
    need_compile = resource->GetResult(kNeedCompile).cast<bool>();
  }
  MS_EXCEPTION_IF_NULL(resource);
  bool is_control_flow = graph_call_condition.is_control_flow_;
  auto fn = [resource, need_compile, is_control_flow, kNeedCompile](const VectorRef &arg_list) -> VectorRef {
    if (need_compile) {
      MS_LOG(DEBUG) << "Start emit action for graph " << resource->func_graph()->ToString();
      auto manager = resource->manager();
      manager->AddFuncGraph(resource->func_graph(), true);
      resource->SetBackendAsync([]() { return compile::CreateBackend(); });
      // kFlagJitCallGraph is set false to compile sub graph in control flow
      if (is_control_flow) {
        for (const auto &g : manager->func_graphs()) {
          g->set_flag(kFlagJitCallGraph, false);
        }
      }
      (void)TaskEmitAction(resource);
      (void)ExecuteAction(resource);
      resource->SetResult(kNeedCompile, false);
    }
    MS_LOG(DEBUG) << "Start execute action for graph " << resource->func_graph()->ToString();
    compile::VmEvalFuncPtr run = resource->GetResult(pipeline::kOutput).cast<compile::VmEvalFuncPtr>();
    return utils::cast<VectorRef>((*run)(arg_list));
  };
  return fn;
}

PrimitivePyPtr AutoGradUtil::BuildBpropCutPrim(const PrimitivePtr &prim, bool is_need_recompute) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_py = prim->cast<PrimitivePyPtr>();
  MS_EXCEPTION_IF_NULL(prim_py);
  auto bprop_cut = std::make_shared<PrimitivePy>("bprop_cut");
  bprop_cut->CopyHookFunction(prim_py);
  prim_py->AddBpropCutPrim(bprop_cut);
  if (prim->HasAttr("cell_id")) {
    auto cell_id = GetValue<std::string>(prim->GetAttr("cell_id"));
    if (!cell_id.empty()) {
      (void)bprop_cut->AddAttr("cell_hook", MakeValue(true));
      (void)bprop_cut->AddAttr("cell_id", MakeValue(cell_id));
    }
  }
  // Only custom op need add this attr, hook function not need.
  if (prim->HasAttr("custom_op_bprop")) {
    (void)bprop_cut->AddAttr("custom_op_bprop", MakeValue(true));
  }
  (void)bprop_cut->AddAttr("custom_op_name", MakeValue(prim->name()));
  if (is_need_recompute) {
    (void)bprop_cut->AddAttr("is_recompute", MakeValue(true));
  }
  return bprop_cut;
}

void AutoGradUtil::CheckRecomputeInputs(const ValuePtrList &inputs, bool is_need_recompute) {
  if (!is_need_recompute) {
    return;
  }
  for (const auto &input : inputs) {
    if (!input->isa<ValueSequence>()) {
      continue;
    }
    const auto &seq = input->cast<ValueSequencePtr>();
    const auto val = seq->value();
    if (NeedGrad(val)) {
      MS_LOG(EXCEPTION) << "For recompute cell, now we do not support calculate tensor's gradient from tuple. "
                           "You need check your inputs of construct function from recompute cell, and not put "
                           "tensors in tuple which need grad!";
    }
  }
}

void AutoGradUtil::ClearAutoGradStaticCache() { jit_call_graph_compile_cache_.clear(); }

void AutoGradUtil::CheckAndSetAbstract(const OpGradInfoPtr &op_grad_info) {
  MS_EXCEPTION_IF_NULL(op_grad_info);
  if (op_grad_info->output_value_simple_info != nullptr) {
    MS_LOG(DEBUG) << "Convert op " << op_grad_info->op_prim->name() << " simple infer info to abstract";
    ConvertSimpleInferInfoToAbstract(op_grad_info);
    return;
  }

  // View op input abs and output abs maybe nullptr
  if (MS_UNLIKELY(op_grad_info->input_abs.empty())) {
    // Get inputs abstract
    MS_LOG(DEBUG) << "Op " << op_grad_info->op_prim->name() << " inputs abstract not set, set it now";
    for (const auto &v : op_grad_info->input_value) {
      // For use abstract cache on tensor
      op_grad_info->input_abs.emplace_back(kGradAbstractConverter.ConvertAbstract(v));
    }
  }
  if (op_grad_info->out_abs == nullptr) {
    MS_LOG(EXCEPTION) << "Get output abs is nullptr";
  }
}

void AutoGradUtil::CacheOutputAbstract(const ValuePtr &v, const abstract::AbstractBasePtr &abs) {
  MS_EXCEPTION_IF_NULL(v);
  MS_EXCEPTION_IF_NULL(abs);

  // Just check size.
  if (v->isa<ValueSequence>()) {
    const auto &value_seq = v->cast<ValueSequencePtr>();
    const auto &abs_seq = abs->cast<abstract::AbstractSequencePtr>();
    if (abs_seq == nullptr) {
      MS_LOG(EXCEPTION) << "Abstract is not abstract sequence, get " << abs->ToString();
    }
    size_t value_size = value_seq->size();
    if (value_size != abs_seq->size()) {
      MS_LOG(EXCEPTION) << "Abstract size " << abs_seq->size() << " is not equal to value size " << value_size;
    }
  }
}

void AutoGradUtil::CheckAndCloneInplaceInput(const kernel::pyboost::OpPtr &inplace_op, const PrimitivePtr &prim,
                                             const std::string &device_target, ValuePtrList &&inputs) {
  auto input_tensor = inputs[0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  ValuePtr val = nullptr;
  if (!kernel::pyboost::OpRunStatus::Get().RequireGrad() ||
      !BpropExpander::IsCloneInplaceInput(BpropCallback(prim, &inputs, &val))) {
    return;
  }
  MS_LOG(DEBUG) << "Begin clone src value for op " << prim->name();
  auto op = CREATE_PYBOOST_OP(Clone, device_target);
  (void)op->Call(input_tensor);
  inplace_op->set_clone_tensor(op->output(0));
}
}  // namespace PyNativeAlgo

bool BpropCallback::IsNotRequiresGrad(size_t index) const {
  // Check Tensor need grad.
  runtime::Pipeline::Get().WaitBpropStage();
  return !PyNativeAlgo::AutoGradUtil::NeedGrad({(*inputs_)[index]});
}

void BpropCallback::FreeDeviceAddress(ValuePtr *value) const {
  *value = PyNativeAlgo::Common::CreateFakeValueWithoutDeviceAddress(*value);
}
}  // namespace pynative
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PIPELINE_PYNATIVE_GRAD_GRAD_UTILS_H_
