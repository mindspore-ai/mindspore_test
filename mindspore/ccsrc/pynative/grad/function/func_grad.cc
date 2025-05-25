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

#include "pynative/grad/function/func_grad.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/primitive_utils.h"
#include "include/common/utils/hook.h"
#include "include/common/pynative/common_utils.h"
#include "pynative/pynative_utils.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "runtime/pipeline/pipeline.h"
#include "pynative/grad/custom_function.h"
#include "pynative/grad/grad_utils.h"
#include "frontend/optimizer/ad/pynative_jit_grad.h"
#include "pynative/grad/primitive_hook.h"
#include "pynative/grad/function_py.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ccsrc/pyboost/functions/auto_grad_guard.h"

namespace mindspore::pynative::autograd {
namespace {
constexpr char kInput[] = "input";

ValuePtr Add(const ValuePtr &input, const ValuePtr &other, const FuncBuilderPtr &func_impl) {
  MS_EXCEPTION_IF_NULL(input);
  MS_EXCEPTION_IF_NULL(other);
  if (input->isa<None>()) {
    return other;
  }
  if (other->isa<None>()) {
    return input;
  }
  auto result = func_impl->Add(input, other);
  MS_EXCEPTION_IF_NULL(result);
  return result;
}

void Add(const ValuePtr &other, size_t input_index, const FuncBuilderPtr &func_impl, std::vector<ValuePtr> *inputs) {
  if (input_index >= inputs->size()) {
    MS_LOG(EXCEPTION) << "The input index should less than inputs size";
  }

  (*inputs)[input_index] = Add(inputs->at(input_index), other, func_impl);
}

ValuePtrList PaddingGradientInput(const ValuePtr &grad, size_t output_size, size_t input_index) {
  ValuePtrList gradients;
  gradients.reserve(output_size);
  for (size_t i = 0; i < output_size; ++i) {
    if (input_index == i) {
      (void)gradients.emplace_back(grad);
    } else {
      // If gradient is not, we just set kNone, then we lazy update zero gradient by
      // LazeUpdateZeroGradient method
      (void)gradients.emplace_back(kNone);
    }
  }
  return gradients;
}

VectorRef GeneratePythonArgs(const OpGradInfoPtr &op_grad_info, const PrimitivePyPtr &prim) {
  VectorRef args;
  size_t input_size = op_grad_info->input_value.size() - op_grad_info->weight_size;
  if (PyNativeAlgo::Common::IsHookNeedSaveInputs(prim)) {
    for (size_t i = 0; i < input_size; ++i) {
      (void)args.emplace_back(op_grad_info->input_value[i]);
    }
    // If we not need recompute, we save output.
    if (!op_grad_info->is_need_recompute) {
      // Hook op can not execute high order now, so we just shallow copy to avoid circle ref.
      (void)args.emplace_back(AutoGradUtil::ShallowCopyAndDetach(op_grad_info->out_value));
    }
  }
  return args;
}

abstract::AbstractBasePtr GenerateFlattenAbs(const ValuePtrList &flatten_values) {
  if (flatten_values.size() == kSizeOne) {
    return CommonUtils::SetAbstractValueToAnyValue(flatten_values[kIndex0]->ToAbstract());
  }
  auto out_value = std::make_shared<ValueTuple>(flatten_values);
  return CommonUtils::SetAbstractValueToAnyValue(out_value->ToAbstract());
}

ValuePtr ValueListToValue(const ValuePtrList &values, const abstract::AbstractBasePtr &abs) {
  if (values.size() == kSizeZero) {
    MS_LOG(EXCEPTION) << "tensors size should not be empty!";
  }
  if (values.size() == kSizeOne && !abs->isa<abstract::AbstractSequence>()) {
    return values[kIndex0];
  }
  return std::make_shared<ValueTuple>(values);
}

bool IsOutputBothEmpty(const ValuePtr &input_grads, const ValuePtr &weight_grads) {
  if (!input_grads->isa<ValueTuple>() || !weight_grads->isa<ValueTuple>()) {
    return false;
  }
  auto input_grads_tuple = input_grads->cast<ValueTuplePtr>();
  auto weight_grads_tuple = weight_grads->cast<ValueTuplePtr>();
  return input_grads_tuple->size() == 0 && weight_grads_tuple->size() == 0;
}

ValuePtr GenerateEmptyTupleValue() {
  std::vector<ValuePtr> value_list;
  auto inputs_value = std::make_shared<ValueTuple>(value_list);
  auto weights_value = std::make_shared<ValueTuple>(value_list);
  std::vector<ValuePtr> tuple_list{inputs_value, weights_value};
  return std::make_shared<ValueTuple>(tuple_list);
}

bool IsNeedComputeGrad(const ValuePtr &input) {
  MS_EXCEPTION_IF_NULL(input);
  if (input->isa<tensor::Tensor>()) {
    const auto &input_tensor = input->cast<tensor::TensorPtr>();
    const auto &auto_grad_meta_data = input_tensor->auto_grad_meta_data();
    if (auto_grad_meta_data == nullptr) {
      return false;
    }
    auto variable = auto_grad_meta_data->UnsafeGetGradNodeImpl();
    if (variable != nullptr) {
      return true;
    }
  } else if (input->isa<ValueSequence>()) {
    auto seq = input->cast<ValueSequencePtr>();
    if (!seq->value().empty() && !seq->value().front()->isa<tensor::Tensor>()) {
      return false;
    }
    return std::any_of(seq->value().begin(), seq->value().end(),
                       [](const ValuePtr &val) { return IsNeedComputeGrad(val); });
  }
  return false;
}

void SetTensorGradMetaData(const ValuePtr &value, const BackwardNodePtr &grad_node, size_t index) {
  auto tensor = value->cast<tensor::TensorPtr>();
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  if (auto_grad_meta_data == nullptr) {
    MS_LOG(DEBUG) << "Tensor " << tensor->id() << " has no auto_grad_meta_data";
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    tensor->set_auto_grad_meta_data(auto_grad_meta_data);
  }
  auto_grad_meta_data->set_grad_node(grad_node);
  auto_grad_meta_data->set_output_index(index);
}

void SetVariable(const ValuePtrList &flatten_outs, const BackwardNodePtr &grad_node) {
  for (size_t i = 0; i < flatten_outs.size(); ++i) {
    if (flatten_outs[i]->isa<tensor::Tensor>()) {
      SetTensorGradMetaData(flatten_outs[i], grad_node, i);
    }
  }
  MS_LOG(DEBUG) << "End update next edge for " << grad_node->ToString();
}

void SetVariableCustom(const ValuePtrList &flatten_inputs, const ValuePtrList &flatten_outs,
                       const BackwardNodePtr &grad_node) {
  for (size_t i = 0; i < flatten_outs.size(); ++i) {
    if (flatten_outs[i]->isa<tensor::Tensor>() && IsNeedComputeGrad(flatten_inputs[i])) {
      SetTensorGradMetaData(flatten_outs[i], grad_node, i);
    }
  }
  MS_LOG(DEBUG) << "End update next edge for " << grad_node->ToString();
}

bool IsValidTensorInput(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  return v->isa<tensor::Tensor>() || v->isa<tensor::MetaSparseTensor>();
}

NodePtrList GenerateNodeInputs(const OpGradInfoPtr &op_grad_info, const FuncBuilderPtr &emitter) {
  NodePtrList node_inputs;
  node_inputs.resize(op_grad_info->input_value.size() + kSizeTwo);
  for (size_t i = 0; i < op_grad_info->input_value.size(); ++i) {
    auto input = op_grad_info->input_value[i];
    if (op_grad_info->clone_value != nullptr && i == kIndex0) {
      // Replace input with clone value.
      // Copy auto grad meta data to avoid need_compute_output flag error.
      input = op_grad_info->clone_value;
    }
    auto func_node = emitter->NewFuncNode(input, op_grad_info->input_abs[i], op_grad_info->input_value_grad_type[i]);
    node_inputs[i] = func_node;
  }
  return node_inputs;
}

void RunPyTensorHook(ValuePtrList *grad_in, const BackwardNodePtr &grad_node) {
  static const std::string kTensorHook = "TensorHook";
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     kTensorHook, false);
  MS_EXCEPTION_IF_NULL(grad_in);
  MS_EXCEPTION_IF_NULL(grad_node);
  runtime::Pipeline::Get().WaitFrontend();
  for (const auto &[hook_id, hook] : grad_node->py_tensor_pre_hooks()) {
    MS_LOG(DEBUG) << "Run hook id T" << hook_id;
    MS_EXCEPTION_IF_NULL(hook);
    (*hook)(grad_in);
  }
  runtime::Pipeline::Get().WaitFrontend();
  MS_LOG(DEBUG) << PyNativeAlgo::Common::PrintDebugInfo(*grad_in, "After hook print gradient in: ");
}

void CallBackwardNodePreHooks(const BackwardNodePtr &grad_node, ValuePtrList *grad_in) {
  MS_EXCEPTION_IF_NULL(grad_in);
  if (!grad_node->py_tensor_pre_hooks().empty()) {
    RunPyTensorHook(grad_in, grad_node);
  }
}

void ReleaseResource(const BackwardNodePtr &grad_node) {
  const auto &forward = PyNativeExecutor::forward_executor();
  if (forward->enable_async()) {
    const auto task = [grad_node]() { grad_node->Release(); };
    runtime::Pipeline::Get().backend_stage()->Push(std::make_shared<BpropTask>(task));
  } else {
    grad_node->Release();
  }
}

void UpdateCreationType(const ValuePtrList &flatten_outputs) {
  for (const auto &output : flatten_outputs) {
    if (output->isa<tensor::Tensor>()) {
      auto output_tensor = output->cast<tensor::TensorPtr>();
      auto view_meta = impl::GetViewAutogradMetaImpl(output_tensor);
      if (view_meta == nullptr) {
        return;
      }
      view_meta->set_creation_type(CreationType::kCustomBprop);
      view_meta->set_version_attr(output_tensor->version().current_version());
    }
  }
}

void CheckInplace(const OpGradInfoPtr &op_grad_info) {
  auto output_tensor = op_grad_info->out_value->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(output_tensor);
  auto view_meta = impl::GetViewAutogradMetaImpl(output_tensor);
  if (view_meta && view_meta->creation_type() != CreationType::kDefault) {
    std::ostringstream ss;
    std::string header = "A view of base is being inplace modified, ";
    ss << header;
    auto grad_node = view_meta->UnsafeGetGradNodeImpl();
    if (view_meta->creation_type() == CreationType::kNoGradMode) {
      ss << "which created in no_grad mode and inplace modified with grad mode enabled.";
      ss << "This case is forbidden, you can put them both in no_grad mode or both in grad enabled.";
    } else if (view_meta->creation_type() == CreationType::kMultiOutput && grad_node) {
      ss << "the " << view_meta->output_index() << "'s  output of " << grad_node->name()
         << "is a view and inplace modified. ";
    }
    if (view_meta->creation_type() == CreationType::kMultiOutput) {
      ss << "This view is one of output for multi output operator, "
         << "which is forbidden. you can use out-of-place op to replace";
    } else if (view_meta->creation_type() == CreationType::kCustomBprop) {
      ss << "This view tensor is output of custom cell, which has custom bprop, it may not support view+inplace, it "
            "will influence grad result,"
         << "you can use out-of-place op to replace";
    }
    MS_LOG(EXCEPTION) << ss.str();
  }
  if (view_meta != nullptr) {
    const auto &base_tensor = view_meta->view_info().base();
    auto auto_grad_meta_data = impl::GetAutogradMetaImpl(base_tensor);
    if (auto_grad_meta_data) {
      auto grad_node = auto_grad_meta_data->UnsafeGetGradNodeImpl();
      if (grad_node != nullptr && isa<LeafNode>(grad_node)) {
        MS_LOG(EXCEPTION) << "A view of leaf tensor that requires grad is being used in an inplace operator, "
                          << op_grad_info->op_prim->name() << ", which is forbidden!";
      }
    }
  }
  auto meta_data = impl::GetAutogradMetaImpl(output_tensor);
  if (meta_data && meta_data->UnsafeGetGradNodeImpl() && isa<LeafNode>(meta_data->UnsafeGetGradNodeImpl())) {
    MS_LOG(EXCEPTION) << "A leaf tensor that requires grad is being used in an inplace operator, "
                      << op_grad_info->op_prim->name() << ", which is forbidden!";
  }
}

void UpdateVersion(const OpGradInfoPtr &op_grad_info, const ValuePtrList &flatten_outputs) {
  if (op_grad_info->operator_type == OperatorType::kDefault) {
    return;
  }
  if (op_grad_info->operator_type == OperatorType::kInplaceOp) {
    AutoGradUtil::BumpVersion(op_grad_info->input_value[kIndex0]);
    return;
  }
  if (op_grad_info->operator_type == OperatorType::kViewOp) {
    for (const auto &output : flatten_outputs) {
      if (output->isa<tensor::Tensor>()) {
        auto out_tensor = output->cast<tensor::TensorPtr>();
        // Op like reshape may partial view.
        if (out_tensor->storage_info() == nullptr) {
          return;
        }
        auto view_meta = impl::GetViewAutogradMetaImpl(out_tensor);
        MS_EXCEPTION_IF_NULL(view_meta);
        view_meta->set_version_attr(out_tensor->version().current_version());
      }
    }
  }
}

void BuildCheckVersionFunc(const OpGradInfoPtr &op_grad_info, const BackwardNodePtr &func,
                           const std::vector<ValuePtr> &flatten_inputs, const std::vector<ValuePtr> &flatten_outputs) {
  std::vector<uint32_t> version_attr;
  std::vector<std::pair<size_t, tensor::Version>> version_with_index;
  auto total_size = flatten_inputs.size() + flatten_outputs.size();
  version_attr.reserve(total_size);
  version_with_index.reserve(total_size);
  for (size_t i = 0; i < flatten_inputs.size(); ++i) {
    const auto &input = flatten_inputs[i];
    if (input->isa<tensor::Tensor>()) {
      const auto flatten_tensor = input->cast<tensor::TensorPtr>();
      if (flatten_tensor->used_in_bprop_graph()) {
        (void)version_with_index.emplace_back(std::make_pair(i, flatten_tensor->version()));
        (void)version_attr.emplace_back(flatten_tensor->version().current_version());
      }
    }
  }
  // We need update version after update next edges, to avoid update variable of inputs.
  // We need collect version of inputs and outputs separately,
  // to avoid that x *= x scene, which not detected.
  UpdateVersion(op_grad_info, flatten_outputs);
  size_t input_size = version_with_index.size();
  for (size_t i = 0; i < flatten_outputs.size(); ++i) {
    const auto &output = flatten_outputs[i];
    if (output->isa<tensor::Tensor>()) {
      const auto flatten_tensor = output->cast<tensor::TensorPtr>();
      if (flatten_tensor->used_in_bprop_graph()) {
        (void)version_with_index.emplace_back(std::make_pair(i, flatten_tensor->version()));
        (void)version_attr.emplace_back(flatten_tensor->version().current_version());
      }
    }
  }
  std::function<void(const std::string &op_name)> check_version_func =
    [inputs = std::move(version_with_index), versions = std::move(version_attr),
     input_size](const std::string &func_name) -> void {
    for (size_t i = 0; i < input_size; ++i) {
      if (inputs[i].second.current_version() != versions[i]) {
        MS_LOG(EXCEPTION)
          << "The " << inputs[i].first << " 's input of " << func_name
          << " has being modified by inplace op, which will cause the gradient error, please check your "
             "inplace operator in code.";
      }
    }
    for (size_t i = input_size; i < inputs.size(); ++i) {
      if (inputs[i].second.current_version() != versions[i]) {
        MS_LOG(EXCEPTION)
          << "The " << inputs[i].first << " 's output of " << func_name
          << " has being modified by inplace op, which will cause the gradient error, please check your "
             "inplace operator in code.";
      }
    }
  };
  func->set_check_func(check_version_func);
}

SavedNodePtr ConstructPlaceHolder(const tensor::TensorPtr &output) {
  auto place_holder = std::make_shared<tensor::Tensor>(*output);
  place_holder->set_device_address(nullptr);
  if (output->storage_info() != nullptr) {
    place_holder->set_storage_info(output->storage_info());
  }
  place_holder->set_used_in_bprop_graph(false);
  place_holder->set_auto_grad_meta_data(nullptr);
  return std::make_shared<SavedNode>(place_holder, nullptr, false, true);
}

// When executing mul->expand_dims->inplace_copy which triggers inplace view operations,
// this constructs a CopySlice node in the graph.
//
// In this scenario:
// - The base tensor (mul's output) will be associated with the CopySlice node
// - CopySlice's inplace_copy function connects to expand_dims via next_edge
// - expand_dims receives an input placeholder that:
//   * Shares autograd metadata with the base tensor
//   * Maintains references through the operation chain
//
// Failing to clear the placeholder's autograd metadata
// will create a reference cycle between these nodes.
void ClearMetaInfofPlaceHolder(const ValuePtrList &flatten_inputs) {
  for (const auto &val : flatten_inputs) {
    if (val->isa<tensor::Tensor>()) {
      const auto &tensor = val->cast<tensor::TensorPtr>();
      if (!tensor->used_in_bprop_graph()) {
        tensor->set_auto_grad_meta_data(nullptr);
      }
    }
  }
}

size_t ProcessDictElement(const ValueDictionaryPtr &dict_value, const ValuePtrList &real_dout, size_t index,
                          VectorRef *args_) {
  MS_EXCEPTION_IF_NULL(args_);
  ValuePtrList key_inputs;
  ValuePtrList value_inputs;
  size_t real_dout_index = index;
  const size_t real_dout_size = real_dout.size();

  for (const auto &elem : dict_value->value()) {
    (void)key_inputs.emplace_back(elem.first);
    if (elem.second->isa<Scalar>()) {
      (void)value_inputs.emplace_back(elem.second);
    } else {
      MS_EXCEPTION_IF_CHECK_FAIL(real_dout_index < real_dout_size, "Real dout out of index, check dict value type.");
      (void)value_inputs.emplace_back(real_dout[real_dout_index++]);
    }
  }
  (void)args_->emplace_back(std::make_shared<ValueTuple>(std::move(key_inputs)));
  (void)args_->emplace_back(std::make_shared<ValueTuple>(std::move(value_inputs)));
  return real_dout_index;
}

void ProcessOutputWithDict(const ValuePtrList &real_dout, size_t index, const ValuePtr &op_output, VectorRef *args_) {
  MS_EXCEPTION_IF_NULL(args_);
  size_t real_dout_index = index;
  const size_t real_dout_size = real_dout.size();
  if (op_output->isa<ValueDictionary>()) {
    const auto &v_dict = op_output->cast<ValueDictionaryPtr>();
    (void)ProcessDictElement(v_dict, real_dout, real_dout_index, args_);
  } else if (op_output->isa<ValueSequence>()) {
    const auto &vec = op_output->cast<ValueSequencePtr>()->value();
    for (const auto &v : vec) {
      if (v->isa<ValueDictionary>()) {
        const auto &v_dict = v->cast<ValueDictionaryPtr>();
        real_dout_index = ProcessDictElement(v_dict, real_dout, real_dout_index, args_);
      } else {
        MS_EXCEPTION_IF_CHECK_FAIL(real_dout_index < real_dout_size, "Real dout out of index, check dict value type.");
        (void)args_->emplace_back(real_dout[real_dout_index++]);
      }
    }
  } else {
    MS_LOG(EXCEPTION) << "Get wrong data type " << op_output->ToString();
  }
}

std::vector<TensorMeta> GenerateInputsMeta(const std::vector<ValuePtr> &inputs) {
  std::vector<TensorMeta> input_meta;
  input_meta.reserve(inputs.size());
  for (const auto &val : inputs) {
    if (val->isa<tensor::Tensor>()) {
      const auto tensor = val->cast<tensor::TensorPtr>();
      (void)input_meta.emplace_back(TensorMeta(tensor->shape(), tensor->Dtype()));
    } else {
      (void)input_meta.emplace_back();
    }
  }
  return input_meta;
}

OpGradInfoPtr GenerateOpGradInfoForCustomFunction(const ValuePtrList &inputs,
                                                  const std::vector<InputType> &input_value_grad_type) {
  OpGradInfoPtr info = std::make_shared<OpGradInfo>();
  info->input_value = inputs;
  info->op_prim = prim::kPrimCellBackwardHook;
  info->input_abs.resize(info->input_value.size());
  for (size_t i = 0; i < info->input_value.size(); i++) {
    info->input_abs[i] = CommonUtils::SetAbstractValueToAnyValue(info->input_value[i]->ToAbstract());
  }

  info->input_value_grad_type = input_value_grad_type;

  return info;
}

void ProcessPost(const ValuePtrList &flatten_outputs, const TensorPtrSet &dirty_tensors,
                 const TensorPtrSet &output_tensors, int num_diff_tensors) {
  if (num_diff_tensors > 1) {
    for (size_t i = 0; i < flatten_outputs.size(); i++) {
      if (flatten_outputs[i]->isa<tensor::Tensor>()) {
        auto base_tensor = flatten_outputs[i]->cast<tensor::TensorPtr>();
        auto view_meta = impl::GetViewAutogradMetaImpl(base_tensor);
        if (view_meta) {
          MS_LOG(DEBUG) << "Set creation type kMultiOutput for tensor " << base_tensor->id();
          view_meta->set_creation_type(CreationType::kMultiOutput);
          view_meta->set_version_attr(base_tensor->version().current_version());
        }
      }
    }
  }

  for (auto &dirty_tensor : dirty_tensors) {
    if (output_tensors.count(dirty_tensor) == 0) {
      MS_LOG(EXCEPTION) << "The dirty tensors must all be outputs of the forward function.";
    }
  }
}

void ProcessForwardOutput(const ValuePtrList &flatten_outputs, const TensorPtrSet &input_base_tensors,
                          const TensorPtrSet &dirty_tensors, const TensorPtrSet &non_diff_tensors,
                          const ValuePtrList &inputs, const std::vector<InputType> &input_value_grad_type,
                          const BackwardNodePtr &grad_node) {
  TensorPtrSet output_tensors;
  int num_diff_tensors = 0;
  for (size_t i = 0; i < flatten_outputs.size(); i++) {
    if (!flatten_outputs[i]->isa<tensor::Tensor>()) {
      continue;
    }
    auto base_tensor = flatten_outputs[i]->cast<tensor::TensorPtr>();
    bool is_input = input_base_tensors.count(base_tensor) > 0;
    bool is_dirty = dirty_tensors.count(base_tensor) > 0;
    bool is_diff = non_diff_tensors.count(base_tensor) == 0;
    MS_LOG(DEBUG) << "Output tensor info, index: " << i << " is_input: " << is_input << " is_dirty: " << is_dirty
                  << " is_diff: " << is_diff;
    if (is_diff) {
      ++num_diff_tensors;
      if (is_dirty) {
        auto meta_data = impl::GetAutogradMetaImpl(base_tensor);
        // tensor is leaf and need grad could not inplace.
        bool is_leaf =
          meta_data && meta_data->UnsafeGetGradNodeImpl() && isa<LeafNode>(meta_data->UnsafeGetGradNodeImpl());
        bool need_grad = AutoGradUtil::NeedGrad(base_tensor);
        MS_LOG(DEBUG) << "Dirty tensor info, index: " << i << " is_leaf: " << is_leaf << " need_grad: " << need_grad;
        if (is_leaf && need_grad) {
          MS_LOG(EXCEPTION) << "A leaf tensor that need grad is being used in an inplace operator.";
        }

        if (!is_input) {
          MS_LOG(WARNING) << "A tensor is not an input, but is given to mark_dirty function.";
        }

        auto view_meta = impl::GetViewAutogradMetaImpl(base_tensor);
        if (view_meta != nullptr && flatten_outputs.size() > 1) {
          MS_LOG(EXCEPTION) << "A view is one of output for multi output operator, "
                            << "which is forbidden. You can use out-of-place op to repalce.";
        }
        // For dirty input tensor, we should rebase variable to new tensor.
        OpGradInfoPtr info = GenerateOpGradInfoForCustomFunction(inputs, input_value_grad_type);
        info->out_value = flatten_outputs[i];
        info->out_abs = GenerateFlattenAbs(flatten_outputs);
        RebaseVariable(info, grad_node, base_tensor, i);
      } else {
        // For the tensor is input and output, we don't need to make a view for it.
        SetTensorGradMetaData(flatten_outputs[i], grad_node, i);
        MS_LOG(DEBUG) << "End update next edge for " << grad_node->ToString();
      }
    }
    auto view_meta = impl::GetViewAutogradMetaImpl(base_tensor);
    if (view_meta && !(is_input && is_dirty)) {
      MS_LOG(DEBUG) << "Set creation type kCustomBprop for tensor " << base_tensor->id();
      view_meta->set_creation_type(CreationType::kCustomBprop);
      view_meta->set_version_attr(base_tensor->version().current_version());
    }

    output_tensors.insert(base_tensor);
  }
  MS_LOG(DEBUG) << "output tensor size: " << output_tensors.size() << " dirty tensor size: " << dirty_tensors.size()
                << "diff tensor num: " << num_diff_tensors;
  ProcessPost(flatten_outputs, dirty_tensors, output_tensors, num_diff_tensors);
}

bool ArgNeedGrad(const BaseRef &arg, const std::unordered_map<BackwardNode *, GradientContext> &gradient_contexts) {
  if (!utils::isa<tensor::Tensor>(arg)) {
    return false;
  }
  const auto &tensor = utils::cast<tensor::TensorPtr>(arg);
  if (tensor->auto_grad_meta_data() == nullptr || tensor->auto_grad_meta_data()->UnsafeGetGradNodeImpl() == nullptr) {
    return false;
  }
  const auto &grad_node = tensor->auto_grad_meta_data()->UnsafeGetGradNodeImpl();
  return gradient_contexts.find(grad_node.get()) != gradient_contexts.end();
}
}  // namespace

void KPynativeOp(const GradParamPtr &grad_param) {
  MS_LOG(DEBUG) << "Begin KPynativeOp"
                << ", prim: " << grad_param->op_grad_info->op_prim->name();
  MS_EXCEPTION_IF_NULL(grad_param);
  const auto &prim = grad_param->op_grad_info->op_prim;
  if (!AutoGradUtil::IsPrimNeedGrad(prim) || !AutoGradUtil::NeedGrad(grad_param->op_grad_info->input_value)) {
    MS_LOG(DEBUG) << "Prim " << prim->name() << " does not need to do op grad.";
    return;
  }
  auto flatten_inputs = CommonUtils::FlattenTensorSeqInValueSeq(grad_param->op_grad_info->input_value);
  BackwardNodePtr fn = nullptr;
  auto flatten_outputs = CommonUtils::FlattenTensorSeqInValue(grad_param->op_grad_info->out_value);
  size_t flatten_output_size = flatten_outputs.size();
  bool is_custom_prim =
    IsPrimitiveEquals(prim, prim::kPrimHookBackward) || IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook);
  if (!is_custom_prim) {
    auto handle = expander::bprop::BpropIRBuilderFactory::Instance().GetBuilder(prim->name());
    if (handle != nullptr) {
      fn = BuildFuncBackwardNode(prim, handle->func, flatten_inputs, grad_param->op_grad_info, flatten_output_size);
    } else {
      fn = BuildCustomBackwardNode(prim, flatten_inputs, grad_param->op_grad_info, flatten_output_size);
    }
  } else {
    grad_param->op_grad_info->out_abs = GenerateFlattenAbs(flatten_outputs);
    fn = BuildHookBackwardNode(prim, flatten_inputs, grad_param->op_grad_info, flatten_output_size);
  }
  if (is_custom_prim) {
    SetVariableCustom(flatten_inputs, flatten_outputs, fn);
    ClearMetaInfofPlaceHolder(flatten_inputs);
    return;
  }
  // Custom hook no need build check func
  BuildCheckVersionFunc(grad_param->op_grad_info, fn, flatten_inputs, flatten_outputs);
  if (grad_param->op_grad_info->operator_type != OperatorType::kInplaceOp) {
    SetVariable(flatten_outputs, fn);
  } else {
    CheckInplace(grad_param->op_grad_info);
    auto output_tensor = grad_param->op_grad_info->out_value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(output_tensor);
    RebaseVariable(grad_param->op_grad_info, fn, output_tensor, kIndex0);
  }
  ClearMetaInfofPlaceHolder(flatten_inputs);
}

bool KPynativeWithFProp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  MS_LOG(DEBUG) << "Do KPynativeWithFProp";
  auto grad_node = BuildGraphBackwardNode(grad_param);
  ValuePtrList flatten_outputs = CommonUtils::FlattenOnlyTensor(grad_param->op_grad_info->out_value);
  SetVariable(flatten_outputs, grad_node);
  MS_LOG(DEBUG) << "End update next edge for " << grad_node->ToString();
  return true;
}

void CallCustomBprop(const CustomContext &context) {
  MS_LOG(DEBUG) << "Begin Call CallCustomBprop";
  if (!AutoGradUtil::NeedGrad(context.inputs)) {
    MS_LOG(DEBUG) << "The custom bprop no need grad!";
    return;
  }
  BackwardNodePtr custom_fn;
  AutoGradUtil::CheckRecomputeInputs(context.inputs, context.is_recompute);
  auto flatten_inputs = CommonUtils::FlattenTensorSeqInValueSeq(context.inputs);
  auto flatten_outputs = CommonUtils::FlattenTensorSeqInValue(context.output);
  UpdateCreationType(flatten_outputs);
  auto input_meta = GenerateInputsMeta(flatten_inputs);
  {
    py::gil_scoped_acquire gil;
    py::list bprop_inputs = context.original_inputs.cast<py::list>();
    SavedNodePtr saved_output = nullptr;
    if (!context.is_recompute) {
      saved_output = SavedNode::ConstructSavedNode(context.output);
    }
    custom_fn = std::make_shared<CustomBackward>("CellCustomBackward", context.bprop_fn, bprop_inputs, saved_output,
                                                 std::move(input_meta), GenerateFlattenAbs(flatten_outputs),
                                                 context.is_recompute, flatten_outputs.size());
  }
  UpdateNextEdges(custom_fn, flatten_inputs);
  SetVariable(flatten_outputs, custom_fn);
  ClearMetaInfofPlaceHolder(flatten_inputs);
  MS_LOG(DEBUG) << "End update next edge for custom bprop, " << custom_fn->ToString();
}

BackwardNodePtr SafeGetGradNodeImpl(const tensor::TensorPtr &tensor) {
  MS_LOG(DEBUG) << "Begin SafeGetGradNodeImpl";
  auto view_meta = impl::GetViewAutogradMetaImpl(tensor);
  if (view_meta == nullptr) {
    auto auto_grad_meta_data = impl::GetAutogradMetaImpl(tensor);
    if (auto_grad_meta_data == nullptr) {
      return nullptr;
    }
    return auto_grad_meta_data->UnsafeGetGradNodeImpl();
  }
  if (tensor->version().current_version() == view_meta->version_attr()) {
    return view_meta->UnsafeGetGradNodeImpl();
  }
  auto handle = expander::bprop::BpropIRBuilderFactory::Instance().GetBuilder("AsStrided");
  std::string device_target = kernel::pyboost::OpRunStatus::Get().device_target();
  auto emitter = std::make_shared<FuncBuilder>("AsStrided", std::move(device_target), nullptr);
  MS_EXCEPTION_IF_NULL(tensor->storage_info());
  auto shape_value = MakeValue(tensor->storage_info()->shape);
  auto strided_value = MakeValue(tensor->storage_info()->strides);
  auto offset_value = MakeValue(tensor->storage_info()->storage_offset);
  // Here we can not set base() as input, it may cause reference cycle.
  auto base_node = emitter->NewFuncNode(CommonUtils::ShallowCopyAndDetach(view_meta->view_info().base()), nullptr,
                                        InputType::kOpOutput);
  auto shape_node = emitter->NewFuncNode(shape_value, nullptr, InputType::kConstant);
  auto strided_node = emitter->NewFuncNode(strided_value, nullptr, InputType::kConstant);
  auto offset_node = emitter->NewFuncNode(offset_value, nullptr, InputType::kConstant);
  // Tow placeholder for out and dout.
  NodePtrList inputs_node{base_node, shape_node, strided_node, offset_node, nullptr, nullptr};
  mindspore::HashMap<std::string, ValuePtr> attrs;
  auto saved_output = ConstructPlaceHolder(tensor);
  auto fn = std::make_shared<FuncBackwardNode>("AsStrided", handle->func, emitter, attrs, inputs_node, saved_output,
                                               nullptr, 1);
  std::vector<ValuePtr> inputs{view_meta->view_info().base(), shape_value, strided_value, offset_value};
  UpdateNextEdges(fn, inputs);
  view_meta->set_grad_node(fn);
  view_meta->set_output_index(kIndex0);
  view_meta->set_version_attr(tensor->version().current_version());
  // To do hook update
  MS_LOG(DEBUG) << "End update next edge for new variable" << fn->ToString();
  return fn;
}

BackwardNodePtr BuildGraphBackwardNode(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  grad_param->is_jit_graph = true;
  auto [cache_hit, bprop_graph] = mindspore::ad::GetBpropGraph(grad_param);
  MS_LOG(DEBUG) << "Bprop Graph cache hit: " << cache_hit;
  bool is_jit_dynamic_shape = grad_param->is_jit_graph;
  // Save replace info in first time
  if (!cache_hit && is_jit_dynamic_shape && grad_param->has_added_v &&
      common::GetCompileConfig("PYNATIVE_JIT_GRAD_MODE") == "1") {
    const auto &jit = PyNativeExecutor::grad_executor()->jit();
    jit->SaveForwardOutputTensorInfoInBpropGraph(bprop_graph, grad_param->graph_cache_key);
  }

  CommonUtils::DumpGraphIR("call_graph.ir", bprop_graph);
  ValuePtrList flatten_outputs;
  CommonUtils::FlattenValueSeqArg(grad_param->op_grad_info->out_value, false, true, &flatten_outputs);
  auto saved_output = SavedNode::ConstructSavedNode(grad_param->op_grad_info->out_value);
  size_t flatten_output_size = flatten_outputs.size();
  auto fn = std::make_shared<GraphBackwardNode>(
    bprop_graph->ToString(), bprop_graph, grad_param->args, grad_param->added_args, saved_output, flatten_output_size,
    grad_param->graph_cache_key, grad_param->is_control_flow, grad_param->is_jit_graph, grad_param->jit_out_has_dict);
  (void)AutoGradUtil::SetValueGradInfo(grad_param->op_grad_info->out_value, InputType::kOpOutput);
  ValuePtrList flatten_inputs =
    CommonUtils::FlattenOnlyTensor(std::make_shared<ValueTuple>(grad_param->op_grad_info->input_value));
  UpdateNextEdges(fn, flatten_inputs);
  return fn;
}

void RebaseVariable(const OpGradInfoPtr &op_grad_info, const BackwardNodePtr &func_node,
                    const tensor::TensorPtr &output_tensor, size_t output_index) {
  auto view_meta = impl::GetViewAutogradMetaImpl(output_tensor);
  if (view_meta != nullptr) {
    MS_LOG(DEBUG) << "Inplace op: " << op_grad_info->op_prim->name()
                  << "'s input is a view tensor, try build copyslice node";
    const auto &base_tensor = view_meta->view_info().base();
    const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    auto emitter = std::make_shared<FuncBuilder>("CopySlice", device_target, nullptr);
    auto copy_slice = std::make_shared<CopySliceNode>("CopySlice", func_node, emitter, 1, base_tensor, output_tensor);
    UpdateNextEdges(copy_slice, {base_tensor});
    for (size_t i = 1; i < func_node->next_edges().size(); ++i) {
      const auto &edge = func_node->next_edges()[i];
      copy_slice->add_next_edge(edge);
    }
    const auto &auto_grad_meta_data = base_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto_grad_meta_data->set_grad_node(copy_slice);
    (void)SafeGetGradNodeImpl(output_tensor);
    // We need set weak_ptr node pf output tensor to inplace func.
    auto grad_node = std::dynamic_pointer_cast<FuncBackwardNode>(func_node);
    MS_EXCEPTION_IF_NULL(grad_node);
    grad_node->set_saved_output(SavedNode::ConstructSavedNode(output_tensor, true));
    MS_LOG(DEBUG) << "End update next edge for " << copy_slice->ToString();
    return;
  }
  // inplace op input tensor is also output tensor.
  auto auto_grad_meta = impl::GetAutogradMetaImpl(output_tensor);
  auto_grad_meta->set_grad_node(func_node);
  auto_grad_meta->set_output_index(output_index);
  MS_LOG(DEBUG) << "End update next edge for " << func_node->ToString();
}

void UpdateNextEdges(const BackwardNodePtr &grad_node, const ValuePtrList &inputs) {
  MS_LOG(DEBUG) << "Get input size " << inputs.size();
  std::vector<Edge> next_edges(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &value = inputs[i];
    if (value->isa<tensor::Tensor>()) {
      const auto &tensor = value->cast<tensor::TensorPtr>();
      auto auto_grad_meta_data = tensor->auto_grad_meta_data();
      // Get scalar tensor
      if (auto_grad_meta_data == nullptr) {
        continue;
      }
      if (auto_grad_meta_data->input_type() == InputType::kParameter && !AutoGradUtil::IsParamRequiresGrad(tensor)) {
        continue;
      }
      auto fn = SafeGetGradNodeImpl(tensor);
      if (fn == nullptr) {
        continue;
      }
      MS_LOG(DEBUG) << "Add next edge for tensor " << tensor->id() << " variable: " << fn->ToString();
      next_edges[i] = Edge(fn, auto_grad_meta_data->output_index());
    }
    // to do sparse tensor.
  }
  grad_node->set_next_edges(std::move(next_edges));
}

ValuePtrList FuncBackwardNode::CallBackward(const ValuePtrList &gradients_in) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     name(), false);
  MS_LOG(DEBUG) << "Begin CallBackward: " << name();
  if (check_func_ != nullptr) {
    check_func_(name());
  }
  PreProcess(gradients_in, emitter_);
  emitter_->SetInputs(name(), &node_inputs_, &attrs_);
  const std::vector<NodePtr> cal_grads_node = grad_func()(emitter_.get());
  ValuePtrList cal_grads_values;
  cal_grads_values.reserve(cal_grads_node.size());
  // Binary op grad result may be nulllptr, we need convert to kNone.
  (void)std::transform(cal_grads_node.begin(), cal_grads_node.end(), std::back_inserter(cal_grads_values),
                       [](const NodePtr &node) -> ValuePtr {
                         if (node == nullptr) {
                           return kNone;
                         }
                         return node->Value();
                       });
  // Set nullptr to avoid reference cycle.
  node_inputs_[node_inputs_.size() - kSizeOne] = nullptr;
  node_inputs_[node_inputs_.size() - kSizeTwo] = nullptr;
  auto gradients = PostProcess(cal_grads_values);
  MS_LOG(DEBUG) << "End CallBackward: " << name();
  return gradients;
}

void FuncBackwardNode::PreProcess(const ValuePtrList &dout, const FuncBuilderPtr &emitter) {
  // The flag of need compute grad should set after pruning graph, because we know whether input of network
  // need grad in grad interface.
  int32_t index = -1;
  for (size_t i = 0; i < node_inputs_.size() - kSizeTwo; ++i) {
    auto value = node_inputs_[i]->Value();
    auto func_node = std::dynamic_pointer_cast<expander::FuncNode>(node_inputs_[i]);
    MS_EXCEPTION_IF_NULL(func_node);
    if (MS_UNLIKELY(index + 1 >= static_cast<int32_t>(next_edges().size()))) {
      MS_LOG(EXCEPTION) << "Index should be less than next edges size, but got " << index + 1 << " vs "
                        << next_edges().size();
    }
    bool is_need_grad = false;
    if (!value->isa<ValueSequence>()) {
      index++;
      is_need_grad = impl::CurrentAutoDiffEngine()->IsInExecGraph(next_edges()[index].grad_node);
    } else {
      auto seq = value->cast<ValueSequencePtr>();
      if (!seq->value().empty() && seq->value()[0]->isa<tensor::Tensor>()) {
        auto begin_index = index;
        index += static_cast<int32_t>(seq->value().size());
        is_need_grad =
          std::any_of(next_edges().begin() + begin_index, next_edges().begin() + index,
                      [](const auto &edge) { return impl::CurrentAutoDiffEngine()->IsInExecGraph(edge.grad_node); });
      } else {
        index++;
        is_need_grad = next_edges()[index].is_defined();
      }
    }
    func_node->set_need_compute_grad_out(is_need_grad);
  }
  auto op_output = saved_output_->Unwrap(shared_from_this());
  node_inputs_[node_inputs_.size() - kSizeTwo] = emitter->NewFuncNode(op_output, out_abs_, InputType::kOpOutput);
  if (dout.size() == kSizeOne && !op_output->isa<ValueSequence>()) {
    node_inputs_[node_inputs_.size() - kSizeOne] = emitter->NewFuncNode(dout[kIndex0], out_abs_, InputType::kOpOutput);
  } else {
    node_inputs_[node_inputs_.size() - kSizeOne] =
      emitter->NewFuncNode(std::make_shared<ValueTuple>(dout), out_abs_, InputType::kOpOutput);
  }
}

ValuePtrList FuncBackwardNode::PostProcess(const ValuePtrList &gradient_value) {
  auto flatten_gradients = CommonUtils::FlattenTensorSeqInValueSeq(gradient_value, false);
  return flatten_gradients;
}

void FuncBackwardNode::Release() {
  for (size_t i = 0; i < node_inputs_.size() - kSizeTwo; ++i) {
    const auto &node = node_inputs_[i];
    MS_EXCEPTION_IF_NULL(node);
    node->SetValue(nullptr);
  }
  check_func_ = nullptr;
  saved_output_ = nullptr;
}

ValuePtrList HookBackwardNode::CallBackward(const ValuePtrList &grads) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     name(), false);
  runtime::Pipeline::Get().WaitFrontend();
  MS_LOG(DEBUG) << "Begin HookBackwardNode CallBackward ";
  auto gradient = ValueListToValue(grads, out_abstract_);
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto func_builder = FuncBuilder(name_, device_target, nullptr);
  // Python grad func can not process None, we need to convert None to zero tensor.
  if (name_ != kCellBackwardHookName) {
    gradient = func_builder.FillZeros(gradient, out_abstract_);
  }
  (void)args_.emplace_back(gradient);
  py::gil_scoped_acquire gil_acquire;
  auto out = RunHookFunction(prim_, args_);
  ValuePtrList gradient_values;
  if (utils::isa<PyObjectRef>(out)) {
    PyObjectRef py_ref = utils::cast<PyObjectRef>(out);
    auto out_py_tuple = py_ref.object_;
    ConvertPyObjectToCTensor(out_py_tuple, &gradient_values, true);
  }
  if (gradient_values.empty()) {
    MS_LOG(EXCEPTION) << "Hook fn output is not <PyObjectRef> type!";
  }
  auto gradient_tensors = PostProcess(gradient_values);
  MS_LOG(DEBUG) << "End HookBackwardNode CallBackward";
  runtime::Pipeline::Get().WaitFrontend();
  return gradient_tensors;
}

void HookBackwardNode::Release() {
  py::gil_scoped_acquire gil;
  prim_ = nullptr;
  args_.clear();
  check_func_ = nullptr;
}

void GraphBackwardNode::SetNeedGradIndexes(
  const std::unordered_map<BackwardNode *, GradientContext> &gradient_contexts) {
  need_grad_indexes_.clear();
  std::transform(args_.begin(), args_.end(), std::back_inserter(need_grad_indexes_),
                 [&gradient_contexts](const auto &arg) { return ArgNeedGrad(arg, gradient_contexts); });
}

ValuePtrList GraphBackwardNode::CallBackward(const ValuePtrList &grads) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     name(), false);
  MS_LOG(DEBUG) << "Begin GraphBackwardNode CallBackward ";
  MS_LOG(DEBUG) << PyNativeAlgo::Common::PrintDebugInfo(grads, "bprop cut input grads: ");
  mindspore::ad::CheckBpropGraphHasInvalidDout(cache_key_, need_grad_indexes_);
  auto graph_call_back = AutoGradUtil::CreateGraphCallBack(func_graph_, cache_key_, graph_call_condition_);
  // Add graph din
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  ValuePtrList flatten_outputs;
  auto op_output = saved_output_->Unwrap(shared_from_this(), true);
  CommonUtils::FlattenValueSeqArg(op_output, false, true, &flatten_outputs);
  auto ir_builder = FuncBuilder(name_, device_target, nullptr);
  auto real_dout = LazeUpdateZeroGradient(grads, &ir_builder, std::make_shared<ValueTuple>(flatten_outputs));
  VectorRef args(args_.begin(), args_.end());
  // If output is jit and has dict output. Key and value will converte into tuples for inputs
  if (!graph_call_condition_.jit_out_has_dict_) {
    for (const auto &arg : real_dout) {
      (void)args.emplace_back(arg);
    }
  } else {
    ProcessOutputWithDict(real_dout, kIndex0, op_output, &args);
  }
  size_t size = args.size();
  if (!added_args_.empty()) {
    args.insert(args.end(), added_args_.begin(), added_args_.end());
  }
  MS_LOG(DEBUG) << "Total args size for bprop graph: " << args.size();
  auto gradient_vec_ref = graph_call_back(args);
  if (kernel::pyboost::OpRunStatus::Get().RequireGrad()) {
    VectorRef input_args(args.begin(), args.begin() + size);
    AutoGradUtil::CreateHighOrderGraph(func_graph_, args, gradient_vec_ref, cache_key_);
  }
  auto gradient_values = common::AnfAlgo::TransformVectorRefToMultiValue(gradient_vec_ref);
  auto gradient_tensors = PostProcess(gradient_values);
  MS_LOG(DEBUG) << "End GraphBackwardNode CallBackward";
  return gradient_tensors;
}

ValuePtrList GraphBackwardNode::LazeUpdateZeroGradient(const ValuePtrList &dout, FuncBuilder *func_builder,
                                                       const ValuePtr &output) {
  if (dout.size() == kSizeOne) {
    return dout;
  }
  ValuePtrList outputs;
  CommonUtils::FlattenValueSeqArg(output, false, false, &outputs);
  if (outputs.size() != dout.size()) {
    MS_LOG(EXCEPTION) << "Gradients size should be same as output size! But got output size: " << outputs.size()
                      << ", gradients size: " << dout.size();
  }
  ValuePtrList real_dout(dout.size());
  for (size_t i = 0; i < dout.size(); ++i) {
    if (dout[i]->isa<None>()) {
      MS_LOG(DEBUG) << "Op " << name() << " has multi outputs, and exist null dout, now do emit zeros";
      auto zero_value =
        AutoGradUtil::BuildSpecialValueGrad(outputs[i], nullptr, func_builder, SpecialType::kZerosLikeType);
      MS_EXCEPTION_IF_NULL(zero_value);
      real_dout[i] = zero_value;
    } else {
      real_dout[i] = dout[i];
    }
  }
  return real_dout;
}

void GraphBackwardNode::Release() {
  func_graph_ = nullptr;
  args_.clear();
  added_args_.clear();
  saved_output_ = nullptr;
}

ValuePtr LeafNode::Zeros(const std::shared_ptr<FuncBuilder> &ib) {
  return ib->Zeros(ib->Value(shape_), ib->Value(static_cast<int64_t>(dtype_->type_id())))->Value();
}

ValuePtrList CopySliceNode::CallBackward(const ValuePtrList &grads) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     name(), false);
  MS_LOG(DEBUG) << "Begin CallBackward: " << name();
  const auto &grad = grads[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(grad);
  auto base_abs = std::make_shared<abstract::AbstractTensor>(base_.dtype(), base_.shape());
  auto grad_node = emitter_->NewFuncNode(grad, base_abs, InputType::kOpOutput);
  ValuePtrList grad_inputs = CallBackwardImpl(grad_node);
  auto gradients = PostProcess(grad_inputs);
  MS_LOG(DEBUG) << "End CallBackward: " << name();
  return gradients;
}

ValuePtrList CopySliceNode::CallBackwardImpl(const NodePtr &grad_node) {
  auto view_offset = output_.storage_offset();
  view_offset = view_offset - base_.storage_offset();
  // To do, replace zeros to empty_strided.
  auto result =
    emitter_->Zeros(emitter_->Value(base_.shape()), emitter_->Value(static_cast<int64_t>(base_.dtype()->type_id())));
  auto clone_grad = emitter_->InplaceCopy(result, grad_node);
  auto grad_slice = emitter_->AsStrided(clone_grad, emitter_->Value(output_.shape()),
                                        emitter_->Value(output_.strides()), emitter_->Value((int64_t)view_offset));
  auto clone_grad_slice = emitter_->Contiguous(grad_slice);
  // If the 0'th child node need grad, we need put the 0'th child node of inplace node in exec graph.
  if (impl::CurrentAutoDiffEngine()->IsInExecGraph(next_edges()[kIndex0].grad_node)) {
    impl::CurrentAutoDiffEngine()->AddNodeToExecGraph(inplace_func_->next_edges()[kIndex0].grad_node);
  }
  auto res = inplace_func_->CallBackward({clone_grad_slice->Value()});
  ValuePtrList grad_inputs(res.size());
  for (size_t i = 0; i < res.size(); ++i) {
    if (i == 0) {
      NodePtr partial_grad;
      emitter_->Zeros(emitter_->Value(base_.shape()), emitter_->Value(static_cast<int64_t>(base_.dtype()->type_id())));
      // The result of inplace func may be nullptr, we need replace with zeros.
      if (res[i] == nullptr || res[i]->isa<None>()) {
        partial_grad = emitter_->Zeros(emitter_->Value(output_.shape()),
                                       emitter_->Value(static_cast<int64_t>(output_.dtype()->type_id())));
        (void)emitter_->InplaceCopy(grad_slice, partial_grad);
      } else {
        partial_grad = emitter_->NewFuncNode(
          res[i], std::make_shared<abstract::AbstractTensor>(output_.dtype(), output_.shape()), InputType::kOpOutput);
      }
      (void)emitter_->InplaceCopy(grad_slice, partial_grad);
      grad_inputs[i] = result->Value();
    } else {
      grad_inputs[i] = res[i];
    }
  }
  return grad_inputs;
}

void CopySliceNode::Release() {
  inplace_func_ = nullptr;
  check_func_ = nullptr;
}

void CallCustomPyFunction(const std::shared_ptr<FunctionContext> &context) {
  MS_LOG(DEBUG) << "Begin Call CallCustomPyFunction";
  if (!AutoGradUtil::NeedGrad(context->inputs)) {
    MS_LOG(DEBUG) << "The custom bprop function no need grad!";
    return;
  }
  BackwardNodePtr custom_fn;
  auto input_meta = GenerateInputsMeta(context->inputs);
  {
    py::gil_scoped_acquire gil;
    custom_fn = std::make_shared<PyBackwardNode>("FunctionCustomBackward", context->backward_fn, context->obj,
                                                 std::move(input_meta), GenerateFlattenAbs(context->flatten_outputs),
                                                 context->flatten_outputs.size());
    py::cast<FunctionPtr>(context->obj)->set_weak_grad_node(custom_fn);
  }
  UpdateNextEdges(custom_fn, context->inputs);
  ProcessForwardOutput(context->flatten_outputs, context->input_base_tensors, context->dirty_tensors,
                       context->non_diff_tensors, context->inputs, context->input_value_grad_type, custom_fn);

  MS_LOG(DEBUG) << "End Call CallCustomPyFunction, " << custom_fn->ToString();
}

void CallCustomCFunction(const ValuePtrList &flatten_outputs, const TensorPtrSet &input_base_tensors,
                         const TensorPtrSet &dirty_tensors, const TensorPtrSet &non_diff_tensors,
                         const ValuePtrList &inputs, const std::vector<InputType> &input_value_grad_type,
                         const BackwardNodePtr &node) {
  UpdateNextEdges(node, inputs);
  for (const auto &dirty_tensor : dirty_tensors) {
    dirty_tensor->BumpVersion();
  }
  ProcessForwardOutput(flatten_outputs, input_base_tensors, dirty_tensors, non_diff_tensors, inputs,
                       input_value_grad_type, node);
}

BackwardNodePtr BuildFuncBackwardNode(const PrimitivePtr &prim, const expander::bprop::BpropBuilderFunc &func,
                                      const ValuePtrList &flatten_inputs, const OpGradInfoPtr &op_grad_info,
                                      size_t flatten_output_size) {
  AutoGradUtil::CheckAndSetAbstract(op_grad_info);
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto emitter = std::make_shared<FuncBuilder>(prim->name(), device_target, nullptr);
  auto node_inputs = GenerateNodeInputs(op_grad_info, emitter);
  auto saved_output = SavedNode::ConstructSavedNode(op_grad_info->out_value);
  auto fn = std::make_shared<FuncBackwardNode>(prim->name(), func, emitter, prim->attrs(), node_inputs, saved_output,
                                               op_grad_info->out_abs, flatten_output_size);
  UpdateNextEdges(fn, flatten_inputs);
  return fn;
}

BackwardNodePtr BuildCustomBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                        const OpGradInfoPtr &op_grad_info, size_t flatten_output_size) {
  AutoGradUtil::CheckAndSetAbstract(op_grad_info);
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(DEBUG) << "Try build custom bprop: " << prim->name();
  {
    py::gil_scoped_acquire gil;
    auto prim_py = prim->cast<PrimitivePyPtr>();
    if (prim_py == nullptr) {
      MS_LOG(DEBUG) << "Prim is not PrimitivePy, can not find python bprop";
      return BuildFakeBackwardNode(prim, flatten_inputs, op_grad_info, flatten_output_size);
    }
    py::function fn = prim_py->GetBpropFunction();
    if (py::isinstance<py::none>(fn)) {
      fn = GetBpropFunction(prim->name());
    }
    if (!fn || py::isinstance<py::none>(fn)) {
      MS_LOG(INFO) << "Can not find bprop function for " << prim->name() << ". fn: " << ConvertPyObjToString(fn);
      return BuildFakeBackwardNode(prim, flatten_inputs, op_grad_info, flatten_output_size);
    }
    (void)prim_py->SetHookFn(fn, HookType::kCustomOpBprop);
  }
  return BuildHookBackwardNode(prim, flatten_inputs, op_grad_info, flatten_output_size);
}

BackwardNodePtr BuildHookBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                      const OpGradInfoPtr &op_grad_info, size_t flatten_output_size) {
  MS_EXCEPTION_IF_NULL(prim);
  auto bprop_cut = AutoGradUtil::BuildBpropCutPrim(prim, op_grad_info->is_need_recompute);
  VectorRef args = GeneratePythonArgs(op_grad_info, bprop_cut);
  // Out abs used for fill zeros, which need be flatten like output.
  auto fn = std::make_shared<HookBackwardNode>(prim->name(), bprop_cut, std::move(args), flatten_output_size,
                                               op_grad_info->out_abs);
  UpdateNextEdges(fn, flatten_inputs);
  return fn;
}

BackwardNodePtr BuildFakeBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                      const OpGradInfoPtr &op_grad_info, size_t flatten_output_size) {
  MS_EXCEPTION_IF_NULL(prim);
  auto fn = std::make_shared<FakeBackwardNode>(prim->name(), flatten_output_size);
  UpdateNextEdges(fn, flatten_inputs);
  return fn;
}

ValuePtr AutoDiff::GetGrads(const ValuePtrList &inputs, const std::vector<BackwardNodePtr> &weights,
                            const std::vector<size_t> &grad_position, const GradAttr &grad_attr) {
  auto inputs_grad = GetInputGrads(inputs, grad_attr.grad_all_inputs, grad_attr.get_by_position, grad_position);
  auto weights_grad = GetWeightGrads(grad_attr.grad_weights, weights, grad_attr.weight_param_is_tuple);
  // Gradients wrt inputs and weights.
  if (inputs_grad != nullptr && weights_grad != nullptr) {
    if (IsOutputBothEmpty(inputs_grad, weights_grad)) {
      return GenerateEmptyTupleValue();
    }
    ValuePtrList gradients{inputs_grad, weights_grad};
    return std::make_shared<ValueTuple>(gradients);
  }
  // Gradients wrt inputs.
  if (inputs_grad != nullptr) {
    return inputs_grad;
  }
  // Gradients wrt weights.
  if (weights_grad != nullptr) {
    return weights_grad;
  }
  // grad_all_inputs, grad_weights and get_by_position are all false.
  if (inputs.empty()) {
    // If no input nodes, return empty tuple.
    return std::make_shared<ValueTuple>(ValuePtrList{});
  }

  // If there are input nodes, return gradient of first input node.
  // Tuple, List, scalar will be ignore
  if (IsValidTensorInput(inputs[kIndex0])) {
    return GetTensorGrad(inputs[kIndex0]);
  }
  MS_LOG(DEBUG) << "Get first input node is not tensor " << inputs[0]->ToString();
  return std::make_shared<ValueTuple>(ValuePtrList{});
}

ValuePtr AutoDiff::GetInputGrads(const ValuePtrList &inputs, bool grad_all_inputs, bool get_by_position,
                                 const std::vector<size_t> &grad_position) {
  std::vector<size_t> grad_pos_list;
  if (get_by_position) {
    grad_pos_list = grad_position;
  } else if (grad_all_inputs) {
    grad_pos_list.resize(inputs.size());
    iota(grad_pos_list.begin(), grad_pos_list.end(), 0);
  } else {
    return nullptr;
  }
  ValuePtrList input_grads;
  input_grads.reserve(inputs.size());
  if (!inputs.empty()) {
    for (size_t index : grad_pos_list) {
      if (index >= inputs.size()) {
        MS_LOG(EXCEPTION) << "Position index " << index << " is exceed input size.";
      }
      // Tuple, List, scalar will be ignore
      if (!IsValidTensorInput(inputs[index])) {
        MS_LOG(DEBUG) << inputs[index]->ToString() << "is no tensor";
        continue;
      }
      (void)input_grads.emplace_back(GetTensorGrad(inputs[index]));
    }
    if (get_by_position && input_grads.size() == kSizeOne) {
      return input_grads[kIndex0];
    }
  }
  return std::make_shared<ValueTuple>(input_grads);
}

ValuePtr AutoDiff::GetTensorGrad(const ValuePtr &val) {
  const auto tensor = PyNativeAlgo::Common::GetTensorFromSparseTensor(val);
  MS_EXCEPTION_IF_NULL(tensor);
  if (const auto grad_node = impl::GetUnsafeGradNodeImpl(tensor)) {
    const auto iter = gradient_contexts_.find(grad_node.get());
    if (iter == gradient_contexts_.end()) {
      MS_LOG(INFO) << "tensor requires grad is true, but not in grad graph";
      const auto leaf_node = std::dynamic_pointer_cast<LeafNode>(grad_node);
      MS_EXCEPTION_IF_NULL(leaf_node);
      return LeafNodeNotInGradButHasTensorHook(leaf_node);
    }
    const auto tensor_grad = iter->second.captured_grad->grad;
    return AutoGradUtil::BuildSpecialValueGrad(tensor, tensor_grad, func_impl_.get(), SpecialType::kZerosLikeType);
  }
  return AutoGradUtil::BuildSpecialValueGrad(val, nullptr, func_impl_.get(), SpecialType::kZerosLikeType);
}

ValuePtr AutoDiff::GetLeafNodeGrad(const BackwardNodePtr &grad_node) {
  MS_EXCEPTION_IF_NULL(grad_node);
  auto leaf_node = std::dynamic_pointer_cast<LeafNode>(grad_node);
  MS_EXCEPTION_IF_NULL(leaf_node);
  auto iter = gradient_contexts_.find(grad_node.get());
  if (iter == gradient_contexts_.end()) {
    MS_LOG(DEBUG) << "tensor participate in forward calculation, but requires_grad is false";
    return leaf_node->Zeros(func_impl_);
  }
  auto tensor_grad = iter->second.captured_grad->grad;
  if (tensor_grad == nullptr) {
    MS_LOG(DEBUG) << "tensor participate in forward calculation, but not need back propagate!";
    return LeafNodeNotInGradButHasTensorHook(leaf_node);
  }
  return tensor_grad;
}

ValuePtr AutoDiff::GetWeightGrads(bool grad_weights, const std::vector<BackwardNodePtr> &weights,
                                  bool weight_param_is_tuple) {
  // No need to return gradient of weights.
  if (!grad_weights) {
    return nullptr;
  }
  if (weight_param_is_tuple) {
    ValuePtrList weight_grads;
    weight_grads.reserve(weights.size());
    for (const auto &weight : weights) {
      (void)weight_grads.emplace_back(GetLeafNodeGrad(weight));
    }
    return std::make_shared<ValueTuple>(weight_grads);
  }
  return GetLeafNodeGrad(weights[0]);
}

ValuePtrList AutoDiff::OnsLike(const ValuePtrList &sens) {
  const auto &v = AutoGradUtil::BuildSpecialValueGrad(std::make_shared<ValueTuple>(sens), nullptr, func_impl_.get(),
                                                      SpecialType::kOnesLikeType);
  auto v_seq = v->cast<ValueTuplePtr>();
  return v_seq->value();
}

void AutoDiff::PruningGradGraph(const ValuePtrList &inputs, const std::vector<BackwardNodePtr> &weights,
                                const GradAttr &grad_attr, const std::vector<size_t> &grad_position) {
  PruningInput(inputs, grad_attr, grad_position);
  PruningWeights(weights, grad_attr);

  // Pruning all node in grad graph
  std::vector<NodeStatus> stack;
  stack.reserve(node_used_in_graph_.size());
  (void)stack.emplace_back(NodeStatus(graph_root_.get(), false));
  std::unordered_set<BackwardNode *> visited;
  visited.reserve(node_used_in_graph_.size());
  while (!stack.empty()) {
    auto &[node, is_processed] = stack.back();
    if (gradient_contexts_.find(node) != gradient_contexts_.end()) {
      // For leaf tensor which is grad barrier.
      stack.pop_back();
      continue;
    }
    if (!is_processed) {
      visited.insert(node);
      is_processed = true;
      for (const auto &next_edge : node->next_edges()) {
        if (!next_edge.is_defined()) {
          continue;
        }
        if (visited.find(next_edge.grad_node.get()) == visited.end()) {
          (void)stack.emplace_back(NodeStatus(next_edge.grad_node.get(), false));
        }
      }
    } else {
      bool need_execute = false;
      for (const auto &next_edge : node->next_edges()) {
        if (!next_edge.is_defined()) {
          continue;
        }
        const auto &it = gradient_contexts_.find(next_edge.grad_node.get());
        if (it != gradient_contexts_.end() && it->second.ShouldExecute()) {
          need_execute = true;
          break;
        }
      }
      if (need_execute) {
        gradient_contexts_[node] = GradientContext(true);
      }
      stack.pop_back();
    }
  }
}

void AutoDiff::ComputeDependencies() {
  std::vector<BackwardNode *> queue{graph_root_.get()};
  while (!queue.empty()) {
    auto node = queue.back();
    queue.pop_back();
    for (const auto &next_edge : node->next_edges()) {
      if (!next_edge.is_defined()) {
        continue;
      }
      const auto &next_node = next_edge.grad_node.get();
      dependencies_[next_node] += 1;
      bool inserted = node_used_in_graph_.insert(next_node).second;
      if (inserted) {
        (void)queue.emplace_back(next_node);
      }
    }
  }
}

void AutoDiff::UpdateDependencies(
  const BackwardNodePtr &root, const mindspore::HashMap<BackwardNode *, ValuePtrList> &input_buffer,
  std::priority_queue<BackwardNodePtr, std::vector<BackwardNodePtr>, CompareNode> *queue,
  std::unordered_map<BackwardNode *, int32_t> *dependencies) {
  std::vector<BackwardNode *> d_queue{root.get()};
  std::unordered_set<BackwardNode *> node_used_in_graph;
  if (input_buffer.count(root.get()) > 0) {
    MS_LOG(DEBUG) << "Node: " << root->name() << " is need calculate after None gradients!";
    queue->push(root);
    return;
  }
  while (!d_queue.empty()) {
    auto node = d_queue.back();
    d_queue.pop_back();
    for (const auto &next_edge : node->next_edges()) {
      if (!next_edge.is_defined()) {
        continue;
      }
      const auto &next_node = next_edge.grad_node.get();
      if (--(*dependencies)[next_node] == 0) {
        if (input_buffer.count(next_node) > 0) {
          MS_LOG(DEBUG) << "Node: " << next_node->name() << " is need calculate after None gradients!";
          queue->push(next_edge.grad_node);
          continue;
        }
        bool inserted = node_used_in_graph.insert(next_node).second;
        if (inserted) {
          (void)d_queue.emplace_back(next_node);
        }
      }
    }
  }
}

std::vector<BackwardNodePtr> AutoDiff::GetWeightsNode(const tensor::TensorPtrList &weights, const ValuePtrList &inputs,
                                                      const GradAttr &grad_attr, bool collect_default_weights) {
  if (collect_default_weights && grad_attr.grad_weights) {
    std::vector<BackwardNodePtr> inputs_node;
    for (const auto &val : inputs) {
      if (val->isa<tensor::Tensor>()) {
        auto tensor = val->cast<tensor::TensorPtr>();
        auto node = impl::GetUnsafeGradNodeImpl(tensor);
        if (node != nullptr) {
          inputs_node.emplace_back(node);
        }
      }
    }
    return GetDefaultWeightsNode(graph_root_, inputs_node);
  }
  if (!grad_attr.grad_weights || weights.empty()) {
    return {};
  }
  std::vector<BackwardNodePtr> weights_node;
  weights_node.reserve(weights.size());
  for (const auto &weight : weights) {
    MS_EXCEPTION_IF_NULL(weight);
    auto auto_grad_meta_data = weight->auto_grad_meta_data();
    if (auto_grad_meta_data == nullptr || auto_grad_meta_data->UnsafeGetGradNodeImpl() == nullptr) {
      MS_LOG(DEBUG) << "weight has not auto grad meta data or grad node!";
      // Fake leaf just for zeros
      (void)weights_node.emplace_back(
        std::make_shared<LeafNode>(weight->param_info() != nullptr ? weight->param_info()->name() : "weight",
                                   weight->shape(), weight->Dtype(), true, false));
      continue;
    }
    (void)weights_node.emplace_back(auto_grad_meta_data->UnsafeGetGradNodeImpl());
  }
  return weights_node;
}

std::vector<BackwardNodePtr> AutoDiff::GetDefaultWeightsNode(const BackwardNodePtr &graph_root,
                                                             const std::vector<BackwardNodePtr> &inputs_node) {
  if (graph_root->IsEmpty()) {
    return {};
  }
  MS_LOG(DEBUG) << "No weights given, try collect weights from graph";
  std::unordered_set<BackwardNode *> visit{};
  std::unordered_set<BackwardNodePtr> inputs_border{inputs_node.begin(), inputs_node.end()};
  std::vector<BackwardNodePtr> duplicate_weights;
  auto compare_node = [](const BackwardNodePtr &lhs, const BackwardNodePtr &rhs) {
    return lhs->seq_id() < rhs->seq_id();
  };
  std::priority_queue<BackwardNodePtr, std::vector<BackwardNodePtr>, decltype(compare_node)> queue(compare_node);
  queue.push(graph_root);
  while (!queue.empty()) {
    auto node = queue.top();
    MS_EXCEPTION_IF_NULL(node);
    queue.pop();
    const bool inserted = visit.insert(node.get()).second;
    bool is_leaf = isa<LeafNode>(node);
    if (!inserted && !is_leaf) {
      continue;
    }
    if (is_leaf && std::dynamic_pointer_cast<LeafNode>(node)->is_parameter()) {
      duplicate_weights.push_back(node);
    }
    if (inputs_border.find(node) != inputs_border.end()) {
      continue;
    }
    for (const auto &next_edge : node->next_edges()) {
      if (!next_edge.is_defined()) {
        continue;
      }
      queue.push(next_edge.grad_node);
    }
  }
  // Remove duplicate node.
  std::vector<BackwardNodePtr> weights;
  weights.reserve(duplicate_weights.size());
  std::unordered_set<BackwardNode *> visit_weight{};
  visit_weight.reserve(duplicate_weights.size());
  for (int64_t i = static_cast<int64_t>(duplicate_weights.size() - 1); i >= 0; i--) {
    const auto &weight = duplicate_weights[i];
    const bool inserted = visit_weight.insert(weight.get()).second;
    if (inserted) {
      (void)weights.emplace_back(weight);
    }
  }
  return weights;
}

void AutoDiff::BackPropagate() {
  MS_LOG(DEBUG) << "Begin BackPropagate";
  std::priority_queue<BackwardNodePtr, std::vector<BackwardNodePtr>, CompareNode> queue;
  queue.emplace(graph_root_);
  mindspore::HashMap<BackwardNode *, ValuePtrList> input_buffer;
  (void)input_buffer.insert({graph_root_.get(), root_gradients_});
  MS_LOG(DEBUG) << "Is running recompute grad " << is_run_recompute_;
  while (!queue.empty()) {
    auto fn = queue.top();
    queue.pop();
    MS_LOG(DEBUG) << "Begin calculate op: " << fn->name() << " gradients!";
    auto ctx_iter = gradient_contexts_.find(fn.get());
    auto gradient_in_iter = input_buffer.find(fn.get());
    if (ctx_iter == gradient_contexts_.end() || gradient_in_iter == input_buffer.end()) {
      MS_LOG(DEBUG) << "No need grad, grad fn is: " << fn->ToString();
      continue;
    }
    auto &gradient_in = gradient_in_iter->second;
    MS_LOG(DEBUG) << PyNativeAlgo::Common::PrintDebugInfo(gradient_in, "Begin print gradient in: ");
    // If register hook by weight, and weight in recomputed cell.So, hook will execute, which is not expect.
    if (!is_run_recompute_ || !isa<LeafNode>(fn)) {
      // to do
      CallBackwardNodePreHooks(fn, &gradient_in);
    }
    if (ctx_iter->second.captured_grad != nullptr) {
      auto tensor_grad = gradient_in[ctx_iter->second.captured_grad->input_index]->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_NULL(tensor_grad);
      ctx_iter->second.captured_grad->SetGradient(tensor_grad);
      continue;
    }
    if (isa<GraphBackwardNode>(fn)) {
      auto graph_backward_node = std::dynamic_pointer_cast<GraphBackwardNode>(fn);
      MS_EXCEPTION_IF_NULL(graph_backward_node);
      graph_backward_node->SetNeedGradIndexes(gradient_contexts_);
    }
    auto gradient_out = fn->CallBackward(gradient_in);
    MS_LOG(DEBUG) << PyNativeAlgo::Common::PrintDebugInfo(gradient_out, "Begin print gradient out: ");
    if (gradient_out.size() < fn->next_edges().size()) {
      MS_LOG(EXCEPTION) << "Fn gradient size should larger than next edges size, but got " << gradient_out.size()
                        << " vs " << fn->next_edges().size()
                        << ". This may because your network has self defined bprop function which args of construct "
                           "function not same as bprop function outputs, please check it";
    }
    for (size_t i = 0; i < fn->next_edges().size(); ++i) {
      const auto &next_edge = fn->next_edges()[i];
      if (!next_edge.is_defined()) {
        continue;
      }
      const auto &last_grad_node = next_edge.grad_node;
      if (gradient_contexts_.find(last_grad_node.get()) == gradient_contexts_.end()) {
        MS_LOG(DEBUG) << "No need grad, grad fn is: " << last_grad_node->ToString();
        continue;
      }
      auto it = dependencies_.find(last_grad_node.get());
      if (MS_UNLIKELY(it == dependencies_.end())) {
        MS_LOG(EXCEPTION) << "Last grad node should be in dependencies!";
      }
      it->second -= 1;
      const auto &last_gradient = gradient_out[i];
      // If last_gradient is None, It represents that this tensor grad is zeros.
      // When we use dependencies, None may cause dependencies count error, so we need update dependencies,
      // Otherwise, some node may not execute!
      if (last_gradient->isa<None>()) {
        if (it->second == 0) {
          UpdateDependencies(last_grad_node, input_buffer, &queue, &dependencies_);
        }
        MS_LOG(DEBUG) << last_grad_node->ToString() << ", its gradient is kNone, no need propagate!";
        // Clear grad node of next edge
        fn->set_next_edge(Edge(), i);
        continue;
      }
      if (it->second == 0) {
        dependencies_.erase(it);
        queue.push(last_grad_node);
      }
      if (input_buffer.find(last_grad_node.get()) != input_buffer.end()) {
        Add(last_gradient, next_edge.input_index, func_impl_, &input_buffer[last_grad_node.get()]);
      } else {
        input_buffer[last_grad_node.get()] =
          PaddingGradientInput(last_gradient, last_grad_node->output_size(), next_edge.input_index);
      }
    }
    (void)input_buffer.erase(fn.get());
    if (!high_order_) {
      ReleaseResource(fn);
    }
  }
  MS_LOG(DEBUG) << "End BackPropagate";
}

ValuePtr AutoDiff::LeafNodeNotInGradButHasTensorHook(const std::shared_ptr<LeafNode> &fn) const {
  MS_EXCEPTION_IF_NULL(fn);
  if (is_run_recompute_ || fn->py_tensor_pre_hooks().empty()) {
    return fn->Zeros(func_impl_);
  }
  ValuePtrList grad_in{};
  (void)grad_in.emplace_back(fn->Zeros(func_impl_));
  RunPyTensorHook(&grad_in, fn);
  auto grad_tensor = grad_in.front()->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(grad_tensor);
  return grad_tensor;
}

void AutoDiff::CheckSensShapeAndType(const ValuePtr &sens_gradient) {
  if (sens_gradient == nullptr) {
    return;
  }
  const auto flatten_sens_gradient = CommonUtils::FlattenOnlyTensor(sens_gradient);
  MS_EXCEPTION_IF_CHECK_FAIL(flatten_sens_out_.size() == flatten_sens_gradient.size(),
                             "The given sens gradient's size should be same as out of network!");
  for (size_t i = 0; i < flatten_sens_out_.size(); ++i) {
    const auto &out_tensor = flatten_sens_out_[i]->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(out_tensor);
    const auto &sens_tensor = flatten_sens_gradient[i]->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(sens_tensor);
    const auto &out_shape = out_tensor->shape();
    const auto &sens_gradient_shape = sens_tensor->shape();
    if (!sens_gradient_shape.empty() && !out_shape.empty()) {
      if (sens_gradient_shape != out_shape) {
        MS_EXCEPTION(ValueError) << "The shape should be " << out_shape << ", but got " << sens_gradient_shape << ", "
                                 << ", sens gradient abs " << sens_tensor->ToAbstract()->ToString() << ", out abs"
                                 << out_tensor->ToAbstract()->ToString();
      }
      const auto &sens_gradient_dtype = sens_tensor->Dtype()->ToString();
      const auto &out_dtype = out_tensor->Dtype()->ToString();
      if (sens_gradient_dtype != out_dtype) {
        MS_EXCEPTION(TypeError) << "The dtype should be " << out_dtype << ", but got " << sens_gradient_dtype << ", "
                                << ", sens gradient abs " << sens_tensor->ToAbstract()->ToString() << ", out abs"
                                << out_tensor->ToAbstract()->ToString();
      }
    }
  }
}

void AutoDiff::BuildGraphRoot(const ValuePtr &sens_gradient, bool has_aux) {
  graph_root_ = std::make_shared<GraphRoot>("GraphRoot");
  if (has_aux) {
    if (!output_->isa<ValueSequence>()) {
      MS_LOG(EXCEPTION)
        << "If you set has_aux for grad or value_and_grad, that forward function should be multi output, but got"
        << output_;
    }
    auto aux_out = output_->cast<ValueSequencePtr>()->value()[0];
    flatten_sens_out_ = CommonUtils::FlattenOnlyTensor(aux_out);
  } else {
    flatten_sens_out_ = CommonUtils::FlattenOnlyTensor(output_);
  }
  if (sens_gradient == nullptr) {
    root_gradients_ = OnsLike(flatten_sens_out_);
  } else {
    root_gradients_ = CommonUtils::FlattenOnlyTensor(sens_gradient);
  }
  if (root_gradients_.size() != flatten_sens_out_.size()) {
    MS_LOG(EXCEPTION) << "Sens size should be same as output, but got" << root_gradients_.size() << " vs "
                      << flatten_sens_out_.size();
  }
  UpdateNextEdges(graph_root_, flatten_sens_out_);
}

void AutoDiff::PruningInput(const ValuePtrList &inputs, const GradAttr &grad_attr,
                            const std::vector<size_t> &grad_position) {
  if (inputs.empty()) {
    return;
  }
  auto set_gradient_context = [this](const ValuePtr &val) {
    auto tensor = PyNativeAlgo::Common::GetTensorFromSparseTensor(val);
    MS_EXCEPTION_IF_NULL(tensor);
    const auto &auto_grad_meta = tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta);
    const auto &grad_node = auto_grad_meta->UnsafeGetGradNodeImpl();
    gradient_contexts_[grad_node.get()] =
      GradientContext(false, std::make_unique<GradientContext::CapturedGradient>(auto_grad_meta->output_index()));
  };

  // Grad all inputs
  if (grad_attr.grad_all_inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (IsValidTensorInput(inputs[i])) {
        MS_LOG(DEBUG) << "Set enable grad for the " << i << "  th input";
        set_gradient_context(inputs[i]);
      }
    }
    return;
  }

  mindspore::HashSet<size_t> grad_pos_list{grad_position.begin(), grad_position.end()};
  // Pruning inputs by position in grad graph
  if (grad_attr.get_by_position) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      if (grad_pos_list.find(i) != grad_pos_list.end() && IsValidTensorInput(inputs[i])) {
        MS_LOG(DEBUG) << "Set enable grad for the " << i << "  th input";
        set_gradient_context(inputs[i]);
      }
    }
    return;
  }

  // Pruning first input in grad graph
  if (!grad_attr.grad_all_inputs && !grad_attr.get_by_position && !grad_attr.grad_weights) {
    if (IsValidTensorInput(inputs[kIndex0])) {
      MS_LOG(DEBUG) << "Set enable grad for the 0 th input";
      set_gradient_context(inputs[kIndex0]);
    }
  }
}

void AutoDiff::PruningWeights(const std::vector<BackwardNodePtr> &weights, const GradAttr &grad_attr) {
  // Pruning weights in grad graph
  if (grad_attr.grad_weights) {
    for (const auto &weight : weights) {
      MS_EXCEPTION_IF_NULL(weight);
      if (isa<LeafNode>(weight) && !std::dynamic_pointer_cast<LeafNode>(weight)->should_execute()) {
        MS_LOG(DEBUG) << "the weight should not back propagate!";
        continue;
      }
      gradient_contexts_[weight.get()] = GradientContext(false, std::make_unique<GradientContext::CapturedGradient>(0));
    }
  }
}

AutoDiff::AutoDiff(const ValuePtr &output, bool high_order, bool is_run_recompute) {
  device_target_ = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  func_impl_ = std::make_shared<FuncBuilder>("func_emitter", device_target_);
  output_ = output;
  is_run_recompute_ = is_run_recompute;
  flatten_sens_out_ = CommonUtils::FlattenOnlyTensor(output);
  high_order_ = high_order;
  MS_LOG(DEBUG) << "Is high order graph: " << high_order_;
}

ValuePtr AutoDiff::RunBackward(const ValuePtrList &inputs, const tensor::TensorPtrList &weights,
                               const std::vector<size_t> &grad_position, const GradAttr &grad_attr,
                               bool collect_default_weights, bool has_aux, const ValuePtr &sens) {
  CheckSensShapeAndType(sens);
  GilReleaseWithCheck gil_release;
  BuildGraphRoot(sens, has_aux);
  std::vector<BackwardNodePtr> weights_node = GetWeightsNode(weights, inputs, grad_attr, collect_default_weights);
  if (graph_root_->IsEmpty()) {
    return GetGrads(inputs, weights_node, grad_position, grad_attr);
  }
  ComputeDependencies();
  PruningGradGraph(inputs, weights_node, grad_attr, grad_position);
  GradFlagGuard grad_flag(high_order_);
  kernel::pyboost::RequireGradGuard requires_grad(high_order_);
  BackPropagate();
  CommonUtils::DumpGraphIR("func_grad.ir", std::make_shared<FuncGraph>());
  if (!is_run_recompute_) {
    python_adapter::PyAdapterCallback::ProcessUnPairedCellHook(true);
  }
  return GetGrads(inputs, weights_node, grad_position, grad_attr);
}

bool AutoDiff::IsInExecGraph(const BackwardNodePtr &node) const {
  if (node == nullptr) {
    return false;
  }
  return gradient_contexts_.find(node.get()) != gradient_contexts_.end();
}

void AutoDiff::AddNodeToExecGraph(const BackwardNodePtr &node) {
  if (gradient_contexts_.find(node.get()) != gradient_contexts_.end()) {
    return;
  }
  gradient_contexts_[node.get()] = GradientContext(true);
}

void AutoDiff::Clear() {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative,
                                     runtime::ProfilerEvent::kPyNativeGradClearAutoGradCell,
                                     runtime::ProfilerRecorder::kNoName, true);
  gradient_contexts_.clear();
  dependencies_.clear();
  node_used_in_graph_.clear();
  flatten_sens_out_.clear();
  root_gradients_.clear();
  output_ = nullptr;
  graph_root_ = nullptr;
}
}  // namespace mindspore::pynative::autograd
