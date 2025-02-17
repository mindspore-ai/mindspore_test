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

#include "pipeline/pynative/grad/function/func_grad.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "pybind_api/gil_scoped_long_running.h"
#include "include/common/utils/primitive_utils.h"
#include "include/common/utils/hook.h"
#include "pipeline/pynative/pynative_utils.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "runtime/pipeline/pipeline.h"
#include "pipeline/pynative/grad/custom_function.h"
#include "pipeline/pynative/grad/grad_utils.h"
#include "frontend/optimizer/ad/pynative_jit_grad.h"

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
      (void)args.emplace_back(op_grad_info->out_value);
    }
  }
  return args;
}

abstract::AbstractBasePtr GenerateFlattenAbs(const ValuePtrList &flatten_values) {
  if (flatten_values.size() == kSizeOne) {
    return PyNativeAlgo::Common::SetAbstractValueToAnyValue(flatten_values[kIndex0]->ToAbstract());
  }
  auto out_value = std::make_shared<ValueTuple>(flatten_values);
  return PyNativeAlgo::Common::SetAbstractValueToAnyValue(out_value->ToAbstract());
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
  if (input->isa<tensor::BaseTensor>()) {
    const auto &input_tensor = input->cast<tensor::BaseTensorPtr>();
    const auto &auto_grad_meta_data = input_tensor->auto_grad_meta_data();
    if (auto_grad_meta_data == nullptr) {
      return false;
    }
    auto variable = auto_grad_meta_data->UnsafeGetVariableImpl();
    if (variable != nullptr && variable->is_need_grad()) {
      return true;
    }
  } else if (input->isa<ValueSequence>()) {
    auto seq = input->cast<ValueSequencePtr>();
    if (!seq->value().empty() && !seq->value().front()->isa<tensor::BaseTensor>()) {
      return false;
    }
    return std::any_of(seq->value().begin(), seq->value().end(),
                       [](const ValuePtr &val) { return IsNeedComputeGrad(val); });
  }
  return false;
}

void SetTensorGradMetaData(const ValuePtr &value, const VariablePtr &variable, size_t index) {
  auto tensor = value->cast<tensor::BaseTensorPtr>();
  auto auto_grad_meta_data = tensor->auto_grad_meta_data();
  if (auto_grad_meta_data == nullptr) {
    MS_LOG(DEBUG) << "Tensor " << tensor->id() << " has no auto_grad_meta_data";
    auto_grad_meta_data = std::make_shared<AutoGradMetaData>();
    tensor->set_auto_grad_meta_data(auto_grad_meta_data);
  }
  auto_grad_meta_data->set_variable(variable);
  auto_grad_meta_data->set_output_index(index);
}

void SetVariable(const ValuePtrList &flatten_outs, const VariablePtr &variable) {
  for (size_t i = 0; i < flatten_outs.size(); ++i) {
    if (flatten_outs[i]->isa<tensor::BaseTensor>()) {
      SetTensorGradMetaData(flatten_outs[i], variable, i);
    }
  }
  MS_LOG(DEBUG) << "End update next edge for " << variable->ToString();
}

void SetVariableCustom(const ValuePtrList &flatten_inputs, const ValuePtrList &flatten_outs,
                       const VariablePtr &variable) {
  for (size_t i = 0; i < flatten_outs.size(); ++i) {
    if (flatten_outs[i]->isa<tensor::BaseTensor>() && IsNeedComputeGrad(flatten_inputs[i])) {
      SetTensorGradMetaData(flatten_outs[i], variable, i);
    }
  }
  MS_LOG(DEBUG) << "End update next edge for " << variable->ToString();
}

bool IsValidTensorInput(const ValuePtr &v) {
  MS_EXCEPTION_IF_NULL(v);
  return v->isa<tensor::BaseTensor>() || v->isa<tensor::MetaSparseTensor>();
}

NodePtrList GenerateNodeInputs(const OpGradInfoPtr &op_grad_info, const FuncBuilderPtr &emitter) {
  NodePtrList node_inputs;
  node_inputs.reserve(op_grad_info->input_value.size() + kSizeFive);
  for (size_t i = 0; i < op_grad_info->input_value.size(); ++i) {
    auto input = op_grad_info->input_value[i];
    if (op_grad_info->clone_value != nullptr && i == kIndex0) {
      // Replace input with clone value.
      // Copy auto grad meta data to avoid need_compute_output flag error.
      auto src_tensor = input->cast<tensor::BaseTensorPtr>();
      MS_EXCEPTION_IF_NULL(src_tensor);
      op_grad_info->clone_value->set_auto_grad_meta_data(src_tensor->auto_grad_meta_data());
      input = op_grad_info->clone_value;
    }
    auto func_node = emitter->NewFuncNode(input, op_grad_info->input_abs[i], op_grad_info->input_value_grad_type[i]);
    (void)node_inputs.emplace_back(func_node);
  }
  (void)node_inputs.emplace_back(
    emitter->NewFuncNode(op_grad_info->out_value, op_grad_info->out_abs, InputType::kOpOutput));
  return node_inputs;
}

AutoGradMetaData *HasTensorHook(const ValuePtr &value) {
  if (value == nullptr) {
    MS_LOG(DEBUG) << "Get null value";
    return nullptr;
  }
  auto tensor = value->cast<tensor::BaseTensorPtr>();
  if (tensor == nullptr) {
    MS_LOG(DEBUG) << "Hook just work on tensor, not support value " << value->ToString();
    return nullptr;
  }
  auto auto_grad_meta = impl::get_autograd_meta_impl(tensor);
  if (auto_grad_meta == nullptr || auto_grad_meta->backward_hooks().empty()) {
    MS_LOG(DEBUG) << "Get empty backward hooks for tensor id " << tensor->id();
    return nullptr;
  }
  return auto_grad_meta.get();
}

void RunTensorHook(ValuePtrList *grad_in, AutoGradMetaData *auto_grad_meta) {
  static const std::string kTensorHook = "TensorHook";
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     kTensorHook, false);
  MS_EXCEPTION_IF_NULL(grad_in);
  MS_EXCEPTION_IF_NULL(auto_grad_meta);
  if (grad_in->size() != kSizeOne) {
    MS_LOG(EXCEPTION) << "Tensor hook just work on one tensor value, not support value sequence";
  }
  runtime::Pipeline::Get().WaitFrontend();
  for (const auto &hook : auto_grad_meta->backward_hooks()) {
    MS_LOG(DEBUG) << "Run hook id T" << hook.first;
    MS_EXCEPTION_IF_NULL(hook.second);
    (*grad_in)[kIndex0] = (*(hook.second))(grad_in->front());
  }
  runtime::Pipeline::Get().WaitFrontend();
  MS_LOG(DEBUG) << PyNativeAlgo::Common::PrintDebugInfo(*grad_in, "After hook print gradient in: ");
}

void CallBackwardHooks(const ValuePtr &value, ValuePtrList *grad_in) {
  MS_EXCEPTION_IF_NULL(grad_in);
  auto auto_grad_meta = HasTensorHook(value);
  if (auto_grad_meta == nullptr) {
    return;
  }
  RunTensorHook(grad_in, auto_grad_meta);
}

void ReleaseResource(const VariablePtr &variable) {
  const auto &forward = PyNativeExecutor::forward_executor();
  if (forward->enable_async()) {
    const auto task = [variable]() { variable->Release(); };
    runtime::Pipeline::Get().backend_stage()->Push(std::make_shared<BpropTask>(task));
  } else {
    variable->Release();
  }
}

void UpdateCreationType(const ValuePtrList &flatten_outputs) {
  for (const auto &output : flatten_outputs) {
    if (output->isa<tensor::BaseTensor>()) {
      auto output_tensor = output->cast<tensor::BaseTensorPtr>();
      auto view_meta = impl::get_view_autograd_meta_impl(output_tensor);
      if (view_meta == nullptr) {
        return;
      }
      view_meta->set_creation_type(CreationType::kCustomBprop);
      view_meta->set_version_attr(output_tensor->version().current_version());
    }
  }
}

void CheckInplace(const OpGradInfoPtr &op_grad_info) {
  auto output_tensor = op_grad_info->out_value->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(output_tensor);
  auto view_meta = impl::get_view_autograd_meta_impl(output_tensor);
  if (view_meta && view_meta->creation_type() != CreationType::kDefault) {
    std::ostringstream ss;
    std::string header = "A view of base is being inplace modified, ";
    ss << header;
    auto variable = view_meta->UnsafeGetVariableImpl();
    if (view_meta->creation_type() == CreationType::kNoGradMode) {
      ss << "which created in no_grad mode and inplace modified with grad mode enabled.";
      ss << "This case is forbidden, you can put them both in no_grad mode or both in grad enabled.";
    } else if (view_meta->creation_type() == CreationType::kMultiOutput && variable) {
      ss << "the " << view_meta->output_index() << "'s  output of " << variable->func_node()->name()
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
    auto auto_grad_meta_data = impl::get_autograd_meta_impl(base_tensor);
    if (auto_grad_meta_data) {
      auto variable = auto_grad_meta_data->UnsafeGetVariableImpl();
      if (variable != nullptr && variable->is_leaf()) {
        MS_LOG(EXCEPTION) << "A view of leaf tensor that requires grad is being used in an inplace operator, "
                          << op_grad_info->op_prim->name() << ", which is forbidden!";
      }
    }
  }
  auto meta_data = impl::get_autograd_meta_impl(output_tensor);
  if (meta_data && meta_data->UnsafeGetVariableImpl() && meta_data->UnsafeGetVariableImpl()->is_leaf()) {
    MS_LOG(EXCEPTION) << "A leaf tensor that requires grad is being used in an inplace operator, "
                      << op_grad_info->op_prim->name() << ", which is forbidden!";
  }
}

void UpdateVersion(const OpGradInfoPtr &op_grad_info, const ValuePtrList &flatten_outputs) {
  if (op_grad_info->operator_type == OperatorType::kDefault) {
    return;
  }
  if (op_grad_info->operator_type == OperatorType::kInplaceOp) {
    PyNativeAlgo::AutoGradUtil::BumpVersion(op_grad_info->input_value[kIndex0]);
    return;
  }
  if (op_grad_info->operator_type == OperatorType::kViewOp) {
    for (const auto &output : flatten_outputs) {
      if (output->isa<tensor::BaseTensor>()) {
        auto out_tensor = output->cast<tensor::BaseTensorPtr>();
        // Op like reshape may partial view.
        if (out_tensor->storage_info() == nullptr) {
          return;
        }
        auto view_meta = impl::get_view_autograd_meta_impl(out_tensor);
        MS_EXCEPTION_IF_NULL(view_meta);
        view_meta->set_version_attr(out_tensor->version().current_version());
      }
    }
  }
}

void BuildCheckVersionFunc(const BackwardNodePtr &func, const std::vector<ValuePtr> &flatten_inputs,
                           const std::vector<ValuePtr> &flatten_outputs) {
  std::vector<uint32_t> version_attr;
  std::vector<std::pair<size_t, tensor::BaseTensorPtr>> input_values_with_index;
  auto total_size = flatten_inputs.size() + flatten_outputs.size();
  version_attr.reserve(total_size);
  input_values_with_index.reserve(total_size);
  for (size_t i = 0; i < flatten_inputs.size(); ++i) {
    const auto &input = flatten_inputs[i];
    if (input->isa<tensor::BaseTensor>()) {
      const auto flatten_tensor = input->cast<tensor::BaseTensorPtr>();
      if (flatten_tensor->used_in_bprop_graph()) {
        (void)input_values_with_index.emplace_back(std::make_pair(i, flatten_tensor));
        (void)version_attr.emplace_back(flatten_tensor->version().current_version());
      }
    }
  }
  size_t input_size = input_values_with_index.size();
  for (size_t i = 0; i < flatten_outputs.size(); ++i) {
    const auto &output = flatten_outputs[i];
    if (output->isa<tensor::BaseTensor>()) {
      const auto flatten_tensor = output->cast<tensor::BaseTensorPtr>();
      if (flatten_tensor->used_in_bprop_graph()) {
        (void)input_values_with_index.emplace_back(std::make_pair(i, flatten_tensor));
        (void)version_attr.emplace_back(flatten_tensor->version().current_version());
      }
    }
  }
  std::function<void(const std::string &op_name)> check_version_func =
    [inputs = std::move(input_values_with_index), versions = std::move(version_attr),
     input_size](const std::string &func_name) -> void {
    for (size_t i = 0; i < input_size; ++i) {
      if (inputs[i].second->version().current_version() != versions[i]) {
        MS_LOG(EXCEPTION)
          << "The " << i << " 's input of " << func_name
          << " has being modified by inplace op, which will cause the gradient error, please check your "
             "inplace operator in code.";
      }
    }
    for (size_t i = input_size; i < inputs.size(); ++i) {
      if (inputs[i].second->version().current_version() != versions[i]) {
        MS_LOG(EXCEPTION)
          << "The " << i - input_size << " 's output of " << func_name
          << " has being modified by inplace op, which will cause the gradient error, please check your "
             "inplace operator in code.";
      }
    }
  };
  func->set_check_func(check_version_func);
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
}  // namespace

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
  auto gradients = PostProcess(cal_grads_values);
  MS_LOG(DEBUG) << "End CallBackward: " << name();
  return gradients;
}

void FuncBackwardNode::PreProcess(const ValuePtrList &dout, const FuncBuilderPtr &emitter) {
  const size_t output_index = node_inputs_.size() - kIndex1;
  const auto &output_node = node_inputs_[output_index];
  const auto &op_output = output_node->Value();
  // The flag of need compute grad should set after pruning graph, because we know whether input of network
  // need grad in grad interface.
  for (size_t i = 0; i < node_inputs_.size() - 1; ++i) {
    auto value = node_inputs_[i]->Value();
    auto func_node = std::dynamic_pointer_cast<expander::FuncNode>(node_inputs_[i]);
    MS_EXCEPTION_IF_NULL(func_node);

    func_node->set_need_compute_grad_out(IsNeedComputeGrad(value));
  }
  if (dout.size() == kSizeOne && !op_output->isa<ValueSequence>()) {
    (void)node_inputs_.emplace_back(emitter->NewFuncNode(dout[kIndex0], output_node->abstract(), InputType::kOpOutput));
  } else {
    (void)node_inputs_.emplace_back(
      emitter->NewFuncNode(std::make_shared<ValueTuple>(dout), output_node->abstract(), InputType::kOpOutput));
  }
}

void FuncBackwardNode::Release() {
  for (const auto &node : node_inputs_) {
    node->SetValue(nullptr);
  }
  check_func_ = nullptr;
}

ValuePtrList HookBackwardNode::CallBackward(const ValuePtrList &grads) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     name(), false);
  runtime::Pipeline::Get().WaitFrontend();
  MS_LOG(DEBUG) << "Begin HookBackwardNode CallBackward ";
  auto gradient = ValueListToValue(grads, out_abstract_);
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  // Python grad func can not process None, we need to convert None to zero tensor.
  auto func_builder = FuncBuilder(name_, device_target, nullptr);
  auto filled_zeros_grad = func_builder.FillZeros(gradient, out_abstract_);
  (void)args_.emplace_back(filled_zeros_grad);
  py::gil_scoped_acquire gil_acquire;
  auto out = prim_->RunHookFunction(args_);
  ValuePtrList gradient_values;
  if (utils::isa<PyObjectRef>(out)) {
    PyObjectRef py_ref = utils::cast<PyObjectRef>(out);
    auto out_py_tuple = py_ref.object_;
    ConvertPyObjectToCTensor(out_py_tuple, &gradient_values, false);
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
}

ValuePtrList GraphBackwardNode::CallBackward(const ValuePtrList &grads) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     name(), false);
  MS_LOG(DEBUG) << "Begin GraphBackwardNode CallBackward ";
  MS_LOG(DEBUG) << PyNativeAlgo::Common::PrintDebugInfo(grads, "bprop cut input grads: ");
  auto graph_call_back =
    PyNativeAlgo::AutoGradUtil::CreateGraphCallBack(func_graph_, cache_key_, graph_call_condition_);
  // Add graph din
  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  ValuePtrList flatten_outputs;
  PyNativeAlgo::DataConvert::FlattenValueSeqArg(op_output_, false, true, &flatten_outputs);
  auto ir_builder = FuncBuilder(name_, device_target, nullptr);
  auto real_dout = LazeUpdateZeroGradient(grads, &ir_builder, std::make_shared<ValueTuple>(flatten_outputs));

  // If output is jit and has dict output. Key and value will converte into tuples for inputs
  if (!graph_call_condition_.jit_out_has_dict_) {
    for (const auto &arg : real_dout) {
      (void)args_.emplace_back(arg);
    }
  } else {
    ProcessOutputWithDict(real_dout, kIndex0, op_output_, &args_);
  }
  if (!added_args_.empty()) {
    args_.insert(args_.end(), added_args_.begin(), added_args_.end());
  }
  MS_LOG(DEBUG) << "Total args size for bprop graph: " << args_.size();
  auto gradient_vec_ref = graph_call_back(args_);
  auto gradient_values = common::AnfAlgo::TransformVectorRefToMultiValue(gradient_vec_ref);
  auto gradient_tensors = PostProcess(gradient_values);
  MS_LOG(DEBUG) << "End GraphBackwardNode CallBackward";
  return gradient_tensors;
}

ValuePtrList CopySliceNode::CallBackward(const ValuePtrList &grads) {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kRunExpanderFunc,
                                     name(), false);
  MS_LOG(DEBUG) << "Begin CallBackward: " << name();
  const auto &grad = grads[0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(grad);
  auto grad_node = emitter_->NewFuncNode(grad, base_->abstract(), InputType::kOpOutput);
  auto view_tensor = op_output_->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(view_tensor);
  NodePtrList grad_inputs = CallBackwardImpl(grad_node, view_tensor);
  ValuePtrList cal_grads_values;
  cal_grads_values.reserve(grad_inputs.size());
  (void)std::transform(grad_inputs.begin(), grad_inputs.end(), std::back_inserter(cal_grads_values),
                       [](const NodePtr &node) -> ValuePtr {
                         if (node == nullptr) {
                           return kNone;
                         }
                         return node->Value();
                       });
  auto gradients = PostProcess(cal_grads_values);
  MS_LOG(DEBUG) << "End CallBackward: " << name();
  return gradients;
}

NodePtrList CopySliceNode::CallBackwardImpl(const NodePtr &grad_node, const tensor::BaseTensorPtr &view_tensor) {
  auto base_tensor = base_->Value()->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(base_tensor);
  MS_EXCEPTION_IF_NULL(view_tensor->storage_info());
  auto view_offset = view_tensor->storage_info()->storage_offset;
  if (base_tensor->storage_info() != nullptr) {
    view_offset = view_offset - base_tensor->storage_info()->storage_offset;
  }
  // To do, replace zeros to empty_strided.
  auto result = emitter_->ZerosLikeExt(base_, emitter_->EmitValue(kNone));
  auto clone_grad = emitter_->InplaceCopy(result, grad_node);
  auto grad_slice =
    emitter_->AsStrided(clone_grad, emitter_->Value(view_tensor->storage_info()->shape),
                        emitter_->Value(view_tensor->storage_info()->strides), emitter_->Value((int64_t)view_offset));
  auto clone_grad_slice = emitter_->Contiguous(grad_slice);
  (void)node_inputs_.emplace_back(clone_grad_slice);
  emitter_->SetInputs(inplace_op_name(), &node_inputs_, &attrs_);
  auto res = inplace_func_(emitter_.get());
  if (res.size() != node_inputs_.size() - kIndex2) {
    MS_LOG(EXCEPTION) << "inplace op gradient size should be same as input, but got " << res.size() << " vs "
                      << node_inputs_.size() - kIndex2;
  }
  NodePtrList grad_inputs(res.size());
  for (size_t i = 0; i < res.size(); ++i) {
    if (i == 0) {
      // The result of inplace func may be nullptr, we need replace with zeros.
      if (res[i] == nullptr || res[i]->Value()->isa<None>()) {
        res[i] = emitter_->ZerosLikeExt(node_inputs_[i], emitter_->EmitValue(kNone));
      }
      (void)emitter_->InplaceCopy(grad_slice, res[i]);
      grad_inputs[i] = result;
    } else {
      grad_inputs[i] = res[i];
    }
  }
  return grad_inputs;
}

void CopySliceNode::Release() {
  for (const auto &node : node_inputs_) {
    node->SetValue(nullptr);
  }
  base_->SetValue(nullptr);
}

FuncGrad::FuncGrad(const ValuePtrList &input_param_values, size_t op_num_in_bprop_graph, bool grad_by_value,
                   bool is_run_recompute) {
  MS_LOG(DEBUG) << "Start FuncGrad, input size: " << input_param_values.size();
  for (size_t i = 0; i < input_param_values.size(); ++i) {
    const auto &input_param_value = input_param_values[i];
    auto func_node = std::make_shared<BackwardNode>(kInput + std::to_string(i));
    auto variable = std::make_shared<FuncVariable>(func_node, true);

    if (!input_param_value->isa<ValueSequence>()) {
      // For hook input
      func_node->set_op_output(input_param_value);
      PyNativeAlgo::AutoGradUtil::SetGradInfoForInputs(input_param_value, variable, &param_meta_grad_info_);
    } else {
      variable->set_is_need_grad(false);
    }
    (void)variable_set_.insert(variable);
    (void)cell_inputs_.emplace_back(input_param_value, variable);
  }
  is_run_recompute_ = is_run_recompute;
  param_meta_grad_info_.reserve(op_num_in_bprop_graph);
  device_target_ = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  func_impl_ = std::make_shared<FuncBuilder>("func_emitter", device_target_);
}

bool FuncGrad::KPynativeOp(const GradParamPtr &grad_param) {
  MS_LOG(DEBUG) << "Begin KPynativeOp"
                << ", prim: " << grad_param->op_grad_info->op_prim->name();
  MS_EXCEPTION_IF_NULL(grad_param);
  auto &prim = grad_param->op_grad_info->op_prim;
  if (!PyNativeAlgo::AutoGradUtil::IsPrimNeedGrad(prim) ||
      (grad_by_value_ && !PyNativeAlgo::AutoGradUtil::NeedGrad(grad_param->op_grad_info->input_value))) {
    MS_LOG(DEBUG) << "Prim " << prim->name() << " does not need to do op grad.";
    return true;
  }
  auto flatten_inputs = PyNativeAlgo::DataConvert::FlattenTensorSeqInValueSeq(grad_param->op_grad_info->input_value);
  ConstructParameterNodes(flatten_inputs);
  BackwardNodePtr fn = nullptr;
  auto flatten_outputs = PyNativeAlgo::DataConvert::FlattenTensorSeqInValue(grad_param->op_grad_info->out_value);
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
  // We need update version after update next edges, to avoid update variable of inputs.
  UpdateVersion(grad_param->op_grad_info, flatten_outputs);
  auto variable = std::make_shared<FuncVariable>(fn, false);
  if (isa<FakeBackwardNode>(fn)) {
    variable->set_is_fake_bprop(true);
    variable->set_fake_prim_name(prim->name());
  }
  variable->set_is_custom_op_variable(is_custom_prim);
  (void)variable_set_.insert(variable);
  if (is_custom_prim) {
    SetVariableCustom(flatten_inputs, flatten_outputs, variable);
    return true;
  }
  // Custom hook no need build check func
  BuildCheckVersionFunc(fn, flatten_inputs, flatten_outputs);
  if (grad_param->op_grad_info->operator_type != OperatorType::kInplaceOp) {
    SetVariable(flatten_outputs, variable);
  } else {
    CheckInplace(grad_param->op_grad_info);
    RebaseVariable(grad_param->op_grad_info, variable);
  }
  return true;
}

void FuncGrad::UpdateOutputNodeOfTopCell(const ValuePtr &sens_out) {
  MS_LOG(DEBUG) << "Real output of top cell is " << PyNativeAlgo::Common::GetIdByValue(sens_out)
                << ", output: " << sens_out->ToString();
  flatten_sens_out_ = PyNativeAlgo::DataConvert::FlattenOnlyTensor(sens_out);
  ConstructParameterNodes(flatten_sens_out_);
}

void FuncGrad::BuildForwardLastNode(const ValuePtr &sens_gradient) {
  if (sens_gradient == nullptr) {
    root_gradients_ = OnsLike(flatten_sens_out_);
  } else {
    root_gradients_ = PyNativeAlgo::DataConvert::FlattenOnlyTensor(sens_gradient);
  }
  auto root = std::make_shared<GraphRoot>("GraphRoot");
  UpdateNextEdges(root, flatten_sens_out_);
  auto sens_variable = std::make_shared<FuncVariable>(root, false);
  if (root_gradients_.empty()) {
    sens_variable->set_is_need_grad(false);
  }
  (void)variable_set_.insert(sens_variable);
  last_variable_ = sens_variable;
}

bool FuncGrad::KPynativeWithFProp(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  MS_LOG(DEBUG) << "Do KPynativeWithFProp";
  if (!grad_by_value_) {
    MS_LOG(EXCEPTION) << "High grad not support pyboost call";
  }
  auto fn = BuildGraphBackwardNode(grad_param);
  auto variable = std::make_shared<FuncVariable>(fn, false);
  (void)variable_set_.insert(variable);
  ValuePtrList flatten_outputs;
  PyNativeAlgo::DataConvert::FlattenValueSeqArg(grad_param->op_grad_info->out_value, false, true, &flatten_outputs);
  SetVariable(flatten_outputs, variable);
  MS_LOG(DEBUG) << "End update next edge for " << variable->ToString();
  return true;
}

void FuncGrad::CallCustomBprop(const CustomContext &context) {
  MS_LOG(DEBUG) << "Begin Call CallCustomBprop";
  BackwardNodePtr custom_fn;
  PyNativeAlgo::AutoGradUtil::CheckRecomputeInputs(context.inputs, context.is_recompute);
  auto flatten_inputs = PyNativeAlgo::DataConvert::FlattenTensorSeqInValueSeq(context.inputs);
  auto flatten_outputs = PyNativeAlgo::DataConvert::FlattenTensorSeqInValue(context.output);
  ConstructParameterNodes(flatten_inputs);
  UpdateCreationType(flatten_outputs);
  {
    py::gil_scoped_acquire gil;
    py::list bprop_inputs = context.original_inputs.cast<py::list>();
    if (!context.is_recompute) {
      bprop_inputs.append(context.original_output);
    }
    custom_fn = std::make_shared<CustomBackward>("CellCustomBackward", context.bprop_fn, bprop_inputs,
                                                 GenerateFlattenAbs(flatten_outputs), context.is_recompute,
                                                 flatten_outputs.size());
  }
  UpdateNextEdges(custom_fn, flatten_inputs);
  auto variable = std::make_shared<FuncVariable>(custom_fn, false);
  variable->set_is_custom_op_variable(true);
  (void)variable_set_.insert(variable);
  SetVariable(flatten_outputs, variable);
  MS_LOG(DEBUG) << "End update next edge for custom bprop, " << variable->ToString();
}

VariablePtr FuncGrad::SafeGetVariableImpl(const tensor::BaseTensorPtr &tensor) {
  MS_LOG(DEBUG) << "Begin SafeGetVariableImpl";
  auto view_meta = impl::get_view_autograd_meta_impl(tensor);
  if (view_meta == nullptr) {
    auto auto_grad_meta_data = impl::get_autograd_meta_impl(tensor);
    if (auto_grad_meta_data == nullptr) {
      return nullptr;
    }
    return auto_grad_meta_data->UnsafeGetVariableImpl();
  }
  if (tensor->version().current_version() == view_meta->version_attr()) {
    return view_meta->UnsafeGetVariableImpl();
  }
  auto handle = expander::bprop::BpropIRBuilderFactory::Instance().GetBuilder("AsStrided");
  auto emitter = std::make_shared<FuncBuilder>("AsStrided", device_target_, nullptr);
  MS_EXCEPTION_IF_NULL(tensor->storage_info());
  auto shape_value = MakeValue(tensor->storage_info()->shape);
  auto strided_value = MakeValue(tensor->storage_info()->strides);
  auto offset_value = MakeValue(tensor->storage_info()->storage_offset);
  auto base_node = emitter->NewFuncNode(view_meta->view_info().base(), nullptr, InputType::kOpOutput);
  auto shape_node = emitter->NewFuncNode(shape_value, nullptr, InputType::kConstant);
  auto strided_node = emitter->NewFuncNode(strided_value, nullptr, InputType::kConstant);
  auto offset_node = emitter->NewFuncNode(offset_value, nullptr, InputType::kConstant);
  auto output_node = emitter->NewFuncNode(tensor, nullptr, InputType::kOpOutput);
  NodePtrList inputs_node{base_node, shape_node, strided_node, offset_node, output_node};
  mindspore::HashMap<std::string, ValuePtr> attrs;
  auto fn = std::make_shared<FuncBackwardNode>("AsStrided", handle->func, emitter, attrs, inputs_node, 1);
  std::vector<ValuePtr> inputs{view_meta->view_info().base(), shape_value, strided_value, offset_value};
  UpdateNextEdges(fn, inputs);
  auto new_variable = std::make_shared<FuncVariable>(fn, false);
  view_meta->set_variable(new_variable);
  view_meta->set_output_index(kIndex0);
  view_meta->set_version_attr(tensor->version().current_version());
  (void)variable_set_.insert(new_variable);
  // To do hook update
  MS_LOG(DEBUG) << "End update next edge for new variable" << new_variable->ToString();
  return new_variable;
}

BackwardNodePtr FuncGrad::BuildGraphBackwardNode(const GradParamPtr &grad_param) {
  MS_EXCEPTION_IF_NULL(grad_param);
  if (ir_bprop_ == nullptr) {
    ir_bprop_ = std::make_unique<IrBprop>(std::make_shared<AdParam>(), device_target_, grad_by_value_);
  }
  grad_param->is_func_grad = true;
  grad_param->is_jit_graph = true;
  auto [cache_hit, bprop_graph] = mindspore::ad::GetBpropGraph(grad_param);
  MS_LOG(DEBUG) << "Bprop Graph cache hit: " << cache_hit;
  bool is_jit_dynamic_shape = grad_param->is_jit_graph && (PyNativeExecutor::grad_executor()->config_no_graph() ||
                                                           grad_param->use_dynamic_shape_process);
  // Save replace info in first time
  if (!cache_hit && is_jit_dynamic_shape && grad_param->has_added_v &&
      common::GetCompileConfig("PYNATIVE_JIT_GRAD_MODE") == "1") {
    const auto &jit = PyNativeExecutor::grad_executor()->jit();
    jit->SaveForwardOutputTensorInfoInBpropGraph(bprop_graph, grad_param->graph_cache_key);
  }

  PyNativeAlgo::Common::DumpGraphIR("call_graph.ir", bprop_graph);
  ValuePtrList flatten_outputs;
  PyNativeAlgo::DataConvert::FlattenValueSeqArg(grad_param->op_grad_info->out_value, false, true, &flatten_outputs);
  size_t flatten_output_size = flatten_outputs.size();
  auto fn = std::make_shared<GraphBackwardNode>(
    bprop_graph->ToString(), bprop_graph, grad_param->args, grad_param->added_args, grad_param->op_grad_info->out_value,
    flatten_output_size, grad_param->graph_cache_key, grad_param->is_control_flow, grad_param->is_jit_graph,
    grad_param->use_dynamic_shape_process, grad_param->jit_out_has_dict);
  (void)PyNativeAlgo::AutoGradUtil::SetValueGradInfo(grad_param->op_grad_info->out_value, InputType::kOpOutput);
  ValuePtrList flatten_inputs;
  PyNativeAlgo::DataConvert::FlattenValueSeqArg(std::make_shared<ValueTuple>(grad_param->op_grad_info->input_value),
                                                false, true, &flatten_inputs);
  ConstructParameterNodes(flatten_inputs);
  UpdateNextEdges(fn, flatten_inputs);
  return fn;
}

void FuncGrad::RebaseVariable(const OpGradInfoPtr &op_grad_info, const VariablePtr &variable) {
  auto input_tensor = op_grad_info->input_value[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto view_meta = impl::get_view_autograd_meta_impl(input_tensor);
  if (view_meta != nullptr) {
    MS_LOG(DEBUG) << "Inplace op: " << op_grad_info->op_prim->name()
                  << "'s input is a view tensor, try build copyslice node";
    auto base_tensor = view_meta->view_info().base();
    auto emitter = std::make_shared<FuncBuilder>("CopySlice", device_target_, nullptr);
    auto handle = expander::bprop::BpropIRBuilderFactory::Instance().GetBuilder(variable->func_node()->name());
    MS_EXCEPTION_IF_NULL(handle);
    auto base_node = emitter->NewFuncNode(base_tensor, nullptr, InputType::kOpOutput);
    base_node->set_need_compute_grad_out(IsNeedComputeGrad(base_tensor));
    auto node_inputs = GenerateNodeInputs(op_grad_info, emitter);
    auto copy_slice =
      std::make_shared<CopySliceNode>("CopySlice", handle->func, op_grad_info->op_prim->attrs(), node_inputs, emitter,
                                      1, base_node, op_grad_info->op_prim->name());
    UpdateNextEdges(copy_slice, {base_tensor});
    for (size_t i = 1; i < variable->func_node()->next_edges().size(); ++i) {
      const auto &edge = variable->func_node()->next_edges()[i];
      copy_slice->add_next_edge(edge);
    }
    auto new_base_variable = std::make_shared<FuncVariable>(copy_slice, false);
    auto auto_grad_meta_data = base_tensor->auto_grad_meta_data();
    MS_EXCEPTION_IF_NULL(auto_grad_meta_data);
    auto_grad_meta_data->set_variable(new_base_variable);
    (void)variable_set_.insert(new_base_variable);
    (void)SafeGetVariableImpl(input_tensor);
    MS_LOG(DEBUG) << "End update next edge for " << new_base_variable->ToString();
    return;
  }
  // inplace op input tensor is also output tensor.
  auto auto_grad_meta = impl::get_autograd_meta_impl(input_tensor);
  auto_grad_meta->set_variable(variable);
  auto_grad_meta->set_output_index(0);
  MS_LOG(DEBUG) << "End update next edge for " << variable->ToString();
}

void FuncGrad::UpdateNextEdges(const BackwardNodePtr &grad_node, const ValuePtrList &inputs) {
  MS_LOG(DEBUG) << "Get input size " << inputs.size();
  std::vector<Edge> next_edges(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const auto &value = inputs[i];
    if (value->isa<tensor::BaseTensor>()) {
      const auto &tensor = value->cast<tensor::BaseTensorPtr>();
      auto auto_grad_meta_data = tensor->auto_grad_meta_data();
      // Get scalar tensor
      if (auto_grad_meta_data == nullptr) {
        continue;
      }
      auto variable = SafeGetVariableImpl(tensor);
      if (variable == nullptr || !variable->is_need_grad()) {
        continue;
      }
      MS_LOG(DEBUG) << "Add next edge for tensor " << tensor->id() << "variable: " << variable->ToString();
      next_edges[i] = Edge(variable, auto_grad_meta_data->output_index());
    }
    // to do sparse tensor.
  }
  grad_node->set_next_edges(std::move(next_edges));
}

void FuncGrad::BackPropagate() {
  MS_LOG(DEBUG) << "Begin BackPropagate";
  const auto &last_node_reverse_iter = GetLastNodeReverseIter();
  const auto &root_fn = (*last_node_reverse_iter)->func_node();
  mindspore::HashMap<BackwardNode *, ValuePtrList> input_buffer;
  (void)input_buffer.insert({root_fn.get(), root_gradients_});
  MS_LOG(DEBUG) << "Is running recompute grad " << is_run_recompute_;
  for (auto iter = last_node_reverse_iter; iter != variable_set_.rend(); ++iter) {
    const auto &variable = *iter;
    const auto &fn = variable->func_node();
    MS_LOG(DEBUG) << "Begin calculate op: " << fn->name() << " gradients!";
    if (!variable->is_need_propagate() || !variable->is_need_grad()) {
      MS_LOG(DEBUG) << "No need grad, variable is: " << variable->ToString();
      WeightNodeNotInGradButHasTensorHook(variable, fn);
      ReleaseResource(variable);
      continue;
    }
    if (static_cast<bool>(MS_UNLIKELY(variable->is_fake_bprop()))) {
      MS_LOG(EXCEPTION) << "Illegal primitive " << variable->fake_prim_name() << "'s bprop not defined";
    }
    auto gradient_in_iter = input_buffer.find(fn.get());
    if (gradient_in_iter == input_buffer.end()) {
      MS_LOG(EXCEPTION) << "Fn not has gradient";
    }
    auto &gradient_in = gradient_in_iter->second;
    MS_LOG(DEBUG) << PyNativeAlgo::Common::PrintDebugInfo(gradient_in, "Begin print gradient in: ");
    // If register hook by weight, and weight in recomputed cell.So, hook will execute, which is not expect.
    if (!is_run_recompute_ || !variable->is_leaf()) {
      CallBackwardHooks(fn->op_output(), &gradient_in);
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
      const auto &last_variable = next_edge.variable;
      // If network not calculates input grad, some op will be pruning, we need skip this op.
      if (!last_variable->is_need_grad()) {
        MS_LOG(DEBUG) << "variable is not need grad, " << last_variable->ToString();
        continue;
      }
      const auto &last_fn = last_variable->func_node();
      const auto &last_gradient = gradient_out[i];
      // If last_gradient is None, It represents that this tensor grad is zeros.
      if (last_gradient->isa<None>()) {
        if (!last_variable->is_custom_op_variable()) {
          MS_LOG(DEBUG) << last_variable->ToString() << ", its gradient is kNone, no need propagate!";
          continue;
        }
        MS_LOG(DEBUG) << "Get custom bprop variable, zeros input din may be have non zeors dout";
      }
      if (input_buffer.find(last_fn.get()) != input_buffer.end()) {
        Add(last_gradient, next_edge.input_index, func_impl_, &input_buffer[last_fn.get()]);
      } else {
        input_buffer[last_fn.get()] =
          PaddingGradientInput(last_gradient, last_fn->output_size(), next_edge.input_index);
      }
      last_variable->set_is_need_propagate(true);
    }
    if (variable->is_leaf()) {
      const auto &grads = input_buffer[fn.get()];
      MS_LOG(DEBUG) << "Get leaf node " << variable->ToString();
      if (grads.empty() || grads[0]->isa<None>()) {
        MS_LOG(EXCEPTION) << variable->ToString() << ", " << (grads.empty() ? "grad is empty" : "grad is kNone");
      }
      auto grad_tensor = grads[0]->cast<tensor::BaseTensorPtr>();
      MS_EXCEPTION_IF_NULL(grad_tensor);
      variable->set_grad(grad_tensor);
    }
    (void)input_buffer.erase(fn.get());
    ReleaseResource(variable);
  }
  MS_LOG(DEBUG) << "End BackPropagate";
}

OrderedSet<FuncVariablePtr>::reverse_iterator FuncGrad::GetLastNodeReverseIter() {
  for (auto iter = variable_set_.rbegin(); iter != variable_set_.rend(); ++iter) {
    if (*iter == last_variable_) {
      last_variable_->set_is_need_propagate(true);
      return iter;
    }
  }
  return variable_set_.rend();
}

void FuncGrad::WeightNodeNotInGradButHasTensorHook(const FuncVariablePtr &variable, const BackwardNodePtr &fn) const {
  if (is_run_recompute_ || !variable->is_leaf() || !HasTensorHook(fn->op_output())) {
    return;
  }
  const auto &v = fn->op_output();
  MS_EXCEPTION_IF_NULL(v);
  if (!v->isa<tensor::BaseTensor>()) {
    return;
  }
  auto tensor = v->cast<tensor::BaseTensorPtr>();
  ValuePtrList grad_in{};
  if (variable->grad() == nullptr) {
    grad_in.emplace_back(func_impl_->Zeros(tensor));
  } else {
    grad_in.emplace_back(variable->grad());
  }
  RunTensorHook(&grad_in, impl::get_autograd_meta_impl(tensor).get());
  auto grad_tensor = grad_in.front()->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(grad_tensor);
  variable->set_grad(grad_tensor);
}

void FuncGrad::ConstructParameterNodes(const ValuePtrList &inputs) {
  for (int32_t i = inputs.size() - 1; i >= 0; i--) {
    auto value = inputs[i];
    if (!value->isa<tensor::BaseTensor>()) {
      continue;
    }
    const auto &tensor = value->cast<tensor::BaseTensorPtr>();
    const auto &auto_grad_meta_data = impl::get_autograd_meta_impl(tensor);
    // Get scalar tensor
    if (auto_grad_meta_data == nullptr || auto_grad_meta_data->UnsafeGetVariableImpl() != nullptr) {
      continue;
    }
    if (PyNativeAlgo::AutoGradUtil::IsParam(auto_grad_meta_data->input_type())) {
      param_meta_grad_info_[tensor] = auto_grad_meta_data;
    }
    if (auto_grad_meta_data->input_type() == InputType::kParameter &&
        PyNativeAlgo::AutoGradUtil::IsParamRequiresGrad(tensor)) {
      auto fn = std::make_shared<BackwardNode>(tensor->param_info()->name());
      fn->set_op_output(value);
      auto variable = std::make_shared<FuncVariable>(fn, true);
      auto_grad_meta_data->set_variable(variable);
      (void)variable_set_.insert(variable);
      weights_used_in_graph_.emplace_back(tensor);
    }
  }
}

BackwardNodePtr FuncGrad::BuildFuncBackwardNode(const PrimitivePtr &prim, const expander::bprop::BpropBuilderFunc &func,
                                                const ValuePtrList &flatten_inputs, const OpGradInfoPtr &op_grad_info,
                                                size_t flatten_output_size) {
  PyNativeAlgo::AutoGradUtil::CheckAndSetAbstract(op_grad_info);
  auto emitter = std::make_shared<FuncBuilder>(prim->name(), device_target_, nullptr);
  auto node_inputs = GenerateNodeInputs(op_grad_info, emitter);
  auto fn =
    std::make_shared<FuncBackwardNode>(prim->name(), func, emitter, prim->attrs(), node_inputs, flatten_output_size);
  UpdateNextEdges(fn, flatten_inputs);
  return fn;
}

BackwardNodePtr FuncGrad::BuildCustomBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                                  const OpGradInfoPtr &op_grad_info, size_t flatten_output_size) {
  PyNativeAlgo::AutoGradUtil::CheckAndSetAbstract(op_grad_info);
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

BackwardNodePtr FuncGrad::BuildHookBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                                const OpGradInfoPtr &op_grad_info, size_t flatten_output_size) {
  MS_EXCEPTION_IF_NULL(prim);
  auto bprop_cut = PyNativeAlgo::AutoGradUtil::BuildBpropCutPrim(prim, op_grad_info->is_need_recompute);
  VectorRef args = GeneratePythonArgs(op_grad_info, bprop_cut);
  // Out abs used for fill zeros, which need be flatten like output.
  auto fn = std::make_shared<HookBackwardNode>(prim->name(), bprop_cut, std::move(args), flatten_output_size,
                                               op_grad_info->out_abs);
  UpdateNextEdges(fn, flatten_inputs);
  return fn;
}

BackwardNodePtr FuncGrad::BuildFakeBackwardNode(const PrimitivePtr &prim, const ValuePtrList &flatten_inputs,
                                                const OpGradInfoPtr &op_grad_info, size_t flatten_output_size) {
  MS_EXCEPTION_IF_NULL(prim);
  auto fn = std::make_shared<FakeBackwardNode>(prim->name(), flatten_output_size);
  UpdateNextEdges(fn, flatten_inputs);
  return fn;
}

ValuePtr FuncGrad::GetGrads(const tensor::BaseTensorPtrList &weights, const std::vector<size_t> &grad_position,
                            const GradAttr &grad_attr) {
  auto inputs_grad = GetInputGrads(grad_attr.grad_all_inputs, grad_attr.get_by_position, grad_position);
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
  if (cell_inputs_.empty()) {
    // If no input nodes, return empty tuple.
    return std::make_shared<ValueTuple>(ValuePtrList{});
  }

  // If there are input nodes, return gradient of first input node.
  // Tuple, List, scalar will be ignore
  if (IsValidTensorInput(cell_inputs_[kIndex0].first)) {
    return PyNativeAlgo::AutoGradUtil::BuildSpecialValueGrad(
      cell_inputs_[kIndex0].first, cell_inputs_[kIndex0].second->grad(), func_impl_.get(), SpecialType::kZerosLikeType);
  }
  MS_LOG(DEBUG) << "Get first input node is not tensor " << cell_inputs_[0].first->ToString();
  return std::make_shared<ValueTuple>(ValuePtrList{});
}

ValuePtr FuncGrad::GetInputGrads(bool grad_all_inputs, bool get_by_position, const std::vector<size_t> &grad_position) {
  std::vector<size_t> grad_pos_list;
  if (get_by_position) {
    grad_pos_list = grad_position;
  } else if (grad_all_inputs) {
    grad_pos_list.resize(cell_inputs_.size());
    iota(grad_pos_list.begin(), grad_pos_list.end(), 0);
  } else {
    return nullptr;
  }
  ValuePtrList input_grads;
  input_grads.reserve(cell_inputs_.size());
  if (!cell_inputs_.empty()) {
    for (size_t index : grad_pos_list) {
      if (index >= cell_inputs_.size()) {
        MS_LOG(EXCEPTION) << "Position index: " << index << " is exceed input size.";
      }
      // Tuple, List, scalar will be ignore
      if (!IsValidTensorInput(cell_inputs_[index].first)) {
        MS_LOG(DEBUG) << cell_inputs_[index].first->ToString() << "is no tensor";
        continue;
      }
      ValuePtr real_dout = PyNativeAlgo::AutoGradUtil::BuildSpecialValueGrad(
        cell_inputs_[index].first, cell_inputs_[index].second->grad(), func_impl_.get(), SpecialType::kZerosLikeType);
      (void)input_grads.emplace_back(real_dout);
    }
    if (get_by_position && input_grads.size() == kSizeOne) {
      return input_grads[kIndex0];
    }
  }
  return std::make_shared<ValueTuple>(input_grads);
}

ValuePtr FuncGrad::GetWeightGrads(bool grad_weights, const tensor::BaseTensorPtrList &weights,
                                  bool weight_param_is_tuple) {
  // No need to return gradient of weights.
  if (!grad_weights) {
    return nullptr;
  }
  if (weight_param_is_tuple) {
    ValuePtrList weight_grads;
    weight_grads.reserve(weights.size());
    for (const auto &weight : weights) {
      (void)weight_grads.emplace_back(GetWeightGrad(weight));
    }
    return std::make_shared<ValueTuple>(weight_grads);
  }
  return GetWeightGrad(weights[0]);
}

ValuePtr FuncGrad::GetWeightGrad(const tensor::BaseTensorPtr &weight) {
  MS_EXCEPTION_IF_NULL(weight);
  auto auto_grad_meta_data = weight->auto_grad_meta_data();
  if (auto_grad_meta_data == nullptr) {
    return func_impl_->Zeros(weight);
  }
  auto variable = auto_grad_meta_data->UnsafeGetVariableImpl();
  const auto &func_variable = std::dynamic_pointer_cast<FuncVariable>(variable);
  MS_LOG(DEBUG) << "Get variable " << (variable != nullptr ? variable->ToString() : "is nullptr");
  if (variable != nullptr && variable->is_need_grad()) {
    // If weight used in the forward network, but requires_grad is false, return zero like.
    if (func_variable->grad() == nullptr ||
        (weight->param_info() != nullptr && !weight->param_info()->requires_grad())) {
      MS_LOG(INFO) << "weight participate in forward calculation, but requires_grad is false";
      return func_impl_->Zeros(weight);
    }
    auto weight_grad = func_variable->grad();
    return weight_grad;
  }
  MS_LOG(INFO) << "weight not participate in forward calculation, but requires grad, id: "
               << PyNativeAlgo::Common::GetIdByValue(weight);
  return func_impl_->Zeros(weight);
}

void FuncGrad::ClearGrads(const tensor::BaseTensorPtrList &weights) {
  // Clear input grads.
  for (const auto &input : cell_inputs_) {
    input.second->set_grad(nullptr);
  }
  cell_inputs_.clear();
}

ValuePtrList FuncGrad::OnsLike(const ValuePtrList &sens) {
  const auto &v = PyNativeAlgo::AutoGradUtil::BuildSpecialValueGrad(std::make_shared<ValueTuple>(sens), nullptr,
                                                                    func_impl_.get(), SpecialType::kOnesLikeType);
  auto v_seq = v->cast<ValueTuplePtr>();
  return v_seq->value();
}

void FuncGrad::CheckSensShapeAndType(const ValuePtr &sens_gradient) {
  if (sens_gradient == nullptr) {
    return;
  }
  const auto flatten_sens_gradient = PyNativeAlgo::DataConvert::FlattenOnlyTensor(sens_gradient);
  MS_EXCEPTION_IF_CHECK_FAIL(flatten_sens_out_.size() == flatten_sens_gradient.size(),
                             "The given sens gradient's size should be same as out of network!");
  for (size_t i = 0; i < flatten_sens_out_.size(); ++i) {
    const auto &out_tensor = flatten_sens_out_[i]->cast<tensor::BaseTensorPtr>();
    MS_EXCEPTION_IF_NULL(out_tensor);
    const auto &sens_tensor = flatten_sens_gradient[i]->cast<tensor::BaseTensorPtr>();
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

void FuncGrad::PruningGradGraph(const tensor::BaseTensorPtrList &weights, const GradAttr &grad_attr,
                                const std::vector<size_t> &grad_position) {
  PruningInput(grad_attr, grad_position);
  PruningWeights(weights, grad_attr);

  // Pruning all node in grad graph
  for (const auto &variable : variable_set_) {
    if (variable->is_leaf()) {
      continue;
    }
    bool is_need_grad =
      std::any_of(variable->func_node()->next_edges().begin(), variable->func_node()->next_edges().end(),
                  [](const auto &edge) { return edge.is_defined() ? edge.variable->is_need_grad() : false; });
    if (!is_need_grad) {
      variable->set_is_need_grad(false);
    }
  }
}

void FuncGrad::PruningInput(const GradAttr &grad_attr, const std::vector<size_t> &grad_position) {
  mindspore::HashSet<size_t> grad_pos_list{grad_position.begin(), grad_position.end()};
  // Pruning inputs by position in grad graph
  if (grad_attr.get_by_position) {
    for (size_t i = 0; i < cell_inputs_.size(); ++i) {
      if (grad_pos_list.find(i) == grad_pos_list.end()) {
        cell_inputs_[i].second->set_is_need_grad(false);
      }
    }
    return;
  }

  // Pruning first input in grad graph
  if (!grad_attr.grad_all_inputs && !grad_attr.get_by_position && !grad_attr.grad_weights) {
    for (size_t i = 1; i < cell_inputs_.size(); ++i) {
      cell_inputs_[i].second->set_is_need_grad(false);
    }
  }

  // Pruning all inputs not grad
  if (!grad_attr.grad_all_inputs && grad_attr.grad_weights) {
    for (auto &cell_input : cell_inputs_) {
      cell_input.second->set_is_need_grad(false);
    }
  }
}

void FuncGrad::PruningWeights(const tensor::BaseTensorPtrList &weights, const GradAttr &grad_attr) {
  // Pruning weights in grad graph
  if (grad_attr.grad_weights) {
    mindspore::HashSet<std::string> grad_weights_id;
    for (const auto &weight : weights) {
      (void)grad_weights_id.emplace(weight->id());
    }
    for (const auto &weight : weights_used_in_graph_) {
      if (grad_weights_id.find(weight->id()) == grad_weights_id.end()) {
        auto variable = weight->auto_grad_meta_data()->UnsafeGetVariableImpl();
        MS_EXCEPTION_IF_NULL(variable);
        variable->set_is_need_grad(false);
      }
    }
  } else {
    for (const auto &weight : weights_used_in_graph_) {
      auto variable = weight->auto_grad_meta_data()->UnsafeGetVariableImpl();
      MS_EXCEPTION_IF_NULL(variable);
      variable->set_is_need_grad(false);
    }
  }
}

ValuePtr FuncGrad::Finish(const tensor::BaseTensorPtrList &weights, const std::vector<size_t> &grad_position,
                          const GradAttr &grad_attr, const ValuePtr &sens) {
  CheckSensShapeAndType(sens);
  GilReleaseWithCheck gil_release;
  BuildForwardLastNode(sens);
  PruningGradGraph(weights, grad_attr, grad_position);
  if (last_variable_->is_need_grad()) {
    BackPropagate();
  }
  PyNativeAlgo::Common::DumpGraphIR("func_grad.ir", std::make_shared<FuncGraph>());
  python_adapter::PyAdapterCallback::ProcessUnPairedCellHook(true);
  ValuePtr gradients = GetGrads(weights, grad_position, grad_attr);
  ClearGrads(weights);
  return gradients;
}
}  // namespace mindspore::pynative::autograd
