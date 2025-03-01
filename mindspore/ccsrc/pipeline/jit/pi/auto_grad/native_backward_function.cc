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
#include "pipeline/jit/pi/auto_grad/native_backward_function.h"
#include <algorithm>
#include <vector>
#include <utility>
#include "include/common/expander/core/node.h"
#include "pyboost/grad_functions/pyboost_grad_functions.h"
#include "include/common/pynative/common_utils.h"

namespace mindspore {
namespace pijit {
namespace grad {
void FuncBuilder::SetInputs(std::string instance_name, const std::vector<NodePtr> *inputs,
                            mindspore::HashMap<std::string, ValuePtr> *attrs_ptr) {
  instance_name_ = std::move(instance_name);
  inputs_ptr_ = inputs;
  attrs_ptr_ = attrs_ptr;
}

ValuePtr FuncBuilder::EmitOp(const PrimitivePtr &prim, const ValuePtrList &inputs) const {
  runtime::ProfilerRecorder profiler(runtime::ProfilerModule::kPynative, runtime::ProfilerEvent::kEmitOp, prim->name(),
                                     false);
  MS_EXCEPTION_IF_NULL(prim);
  MS_LOG(DEBUG) << "Emit op " << prim->name();
  abstract::AbstractBasePtrList input_abs;
  input_abs.reserve(inputs.size());
  std::vector<InputType> input_mask;
  input_mask.reserve(inputs.size());
  for (const auto &input : inputs) {
    auto abs = input->ToAbstract();
    (void)input_abs.emplace_back(abs);
    (void)input_mask.emplace_back(InputType::kInput);
  }
  VectorRef outputs;
  runtime::OpRunnerInfo op_runner_info{prim, device_target_, inputs, input_abs, input_mask, nullptr};
  runtime::PyBoostOpExecute::GetInstance().Execute(&op_runner_info, &outputs);
  auto real_outputs = common::AnfAlgo::TransformVectorRefToMultiValue(outputs);
  if (op_runner_info.output_value_simple_info != nullptr) {
    // Get output abstract
    op_runner_info.output_abs = TransformValueSimpleInfoToAbstract(*op_runner_info.output_value_simple_info);
  }
  MS_EXCEPTION_IF_NULL(op_runner_info.output_abs);
  if (real_outputs.size() == kSizeOne && !op_runner_info.output_abs->isa<abstract::AbstractSequence>()) {
    return real_outputs[kIndex0];
  }
  return std::make_shared<ValueTuple>(std::move(real_outputs));
}

NativeBackwardFuncPtr NativeBackwardFunc::GetInstance(const PrimitivePtr &prim) {
  if (prim == nullptr) {
    return nullptr;
  }
  const auto handle = expander::bprop::BpropIRBuilderFactory::Instance().GetBuilder(prim->name());
  if (handle == nullptr) {
    return nullptr;
  }
  const FuncBuilderPtr &ir_builder = std::make_shared<FuncBuilder>(prim->name());
  return std::make_shared<NativeBackwardFunc>(prim, ir_builder, handle);
}

ValuePtrList NativeBackwardFunc::Run(const ValuePtrList &inputs, const ValuePtr &out, const ValuePtr &dout) {
  if (handle_ == nullptr) {
    return ValuePtrList(GetGradientIndexes().size(), kNone);
  }
  mindspore::HashMap<std::string, ValuePtr> attrs = prim_->attrs();
  NodePtrList node_inputs = PreProcess(inputs, out, dout);
  ir_builder_->SetInputs(GetName(), &node_inputs, &attrs);
  const std::vector<NodePtr> cal_grads_node = handle_->func(ir_builder_.get());
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
  return PostProcess(pynative::CommonUtils::FlattenTensorSeqInValueSeq(cal_grads_values));
}

ValuePtrList NativeBackwardFunc::PostProcess(const ValuePtrList &gradient_value) {
  ValuePtrList grad_values;
  (void)std::transform(GetGradientIndexes().begin(), GetGradientIndexes().end(), std::back_inserter(grad_values),
                       [&gradient_value](const auto &index) -> ValuePtr { return gradient_value[index]; });
  return grad_values;
}

InputType GetInputType(const ValuePtr &input) {
  if (input->template isa<Parameter>()) {
    return InputType::kParameter;
  }
  if (!input->template isa<tensor::Tensor>()) {
    return InputType::kConstant;
  }
  return InputType::kInput;
}

NodePtrList NativeBackwardFunc::PreProcess(const ValuePtrList &inputs, const ValuePtr &out,
                                           const ValuePtr &dout) const {
  NodePtrList node_inputs;
  (void)std::transform(inputs.begin(), inputs.end(), std::back_inserter(node_inputs), [this](const auto &input) {
    if (input == nullptr) {
      return ir_builder_->NewFuncNode(kNone, kNone->ToAbstract(), InputType::kConstant);
    }
    ValuePtr value = input;
    if (input->template isa<stub::TensorNode>()) {
      value = input->template cast<stub::StubNodePtr>()->WaitValue();
    }
    return ir_builder_->NewFuncNode(value, value->ToAbstract(), GetInputType(value));
  });
  std::for_each(GetGradientIndexes().begin(), GetGradientIndexes().end(), [&node_inputs](const auto &index) {
    std::dynamic_pointer_cast<expander::FuncNode>(node_inputs[index])->set_need_compute_grad_out(true);
  });
  (void)node_inputs.emplace_back(ir_builder_->NewFuncNode(out, out->ToAbstract(), InputType::kOpOutput));
  (void)node_inputs.emplace_back(ir_builder_->NewFuncNode(dout, dout->ToAbstract(), InputType::kOpOutput));
  return node_inputs;
}
}  // namespace grad
}  // namespace pijit
}  // namespace mindspore
