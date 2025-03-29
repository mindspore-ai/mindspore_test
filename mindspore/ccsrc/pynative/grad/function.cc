/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/common/utils/convert_utils_py.h"
#include "include/common/utils/tensor_py.h"
#include "pynative/pynative_execute.h"
#include "pynative/pynative_utils.h"
#include "pynative/grad/grad_utils.h"
#include "pynative/grad/function.h"

namespace mindspore::pynative::autograd {
void PrepareForForward() {
  runtime::Pipeline::Get().WaitFrontend();

  const auto &pynative_executor = PyNativeExecutor::GetInstance();
  MS_EXCEPTION_IF_NULL(pynative_executor);
  kernel::pyboost::OpStatus status{false, false, pynative_executor->grad_executor()->custom_bprop_cell_count(),
                                   pynative_executor->forward_executor()->device_target()};
  kernel::pyboost::OpRunStatus::Get().set_run_info(std::move(status));
}

void AutogradContext::MarkDirty(const BaseTensorPtrList &inputs) {
  dirty_inputs_.clear();
  dirty_inputs_.reserve(inputs.size());
  for (const auto &input : inputs) {
    dirty_inputs_.insert(input);
  }
}

void AutogradContext::MarkNonDifferentiable(const BaseTensorPtrList &outputs) {
  non_differentiable_.clear();
  non_differentiable_.reserve(outputs.size());
  for (const auto &output : outputs) {
    non_differentiable_.insert(output);
  }
}

bool AutogradContext::NeedsInputGrad(size_t tensor_index) const {
  const auto &node = node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  const auto &edge_list = node->next_edges();
  MS_EXCEPTION_IF_CHECK_FAIL(tensor_index < edge_list.size(), "tensor index out of range");
  const auto &edge = edge_list[tensor_index];
  return edge.is_defined() ? edge.variable->is_need_grad() : false;
}

bool AutogradContext::NeedGrad(const BaseTensorPtr &tensor) {
  runtime::Pipeline::Get().WaitBpropStage();
  return PyNativeAlgo::AutoGradUtil::NeedGrad(tensor);
}

void CppFunctionDoGrad(AutogradContext *context, const BaseTensorPtrList &inputs, BaseTensorPtrList *outputs) {
  auto node = context->node_.lock();
  MS_EXCEPTION_IF_NULL(node);
  const auto &function_name = node->name();
  const auto &pynative_executor = PyNativeExecutor::GetInstance();
  const auto &grad_executor = pynative_executor->grad_executor();
  MS_EXCEPTION_IF_NULL(pynative_executor);
  if (GradState::Get().RequiresGrad()) {
    MS_LOG(DEBUG) << function_name << " Begin build grad graph";

    // process input
    BaseTensorPtrSet input_tensor_set;
    ValuePtrList input_value_list;
    std::vector<InputType> input_value_grad_type;
    input_value_list.reserve(inputs.size());
    input_value_grad_type.reserve(inputs.size());
    for (const auto input_tensor : inputs) {
      MS_EXCEPTION_IF_NULL(input_tensor);
      (void)input_value_list.emplace_back(input_tensor);
      (void)input_value_grad_type.emplace_back(
        PyNativeAlgo::AutoGradUtil::SetValueGradInfo(input_tensor, InputType::kConstant));
      (void)input_tensor_set.insert(input_tensor);
    }

    // process no diff
    ValuePtrList flatten_outputs_value;
    flatten_outputs_value.reserve(outputs->size());
    for (auto &output_tensor : *outputs) {
      flatten_outputs_value.emplace_back(output_tensor);

      const bool is_diff = context->non_differentiable_.count(output_tensor) == 0;
      const bool is_input = input_tensor_set.count(output_tensor);
      if (!is_diff && is_input) {
        output_tensor = std::make_shared<tensor::Tensor>(*output_tensor);
        output_tensor->set_auto_grad_meta_data(nullptr);
      } else {
        (void)PyNativeAlgo::AutoGradUtil::SetValueGradInfo(output_tensor, InputType::kOpOutput);
      }
    }

    // Do grad
    if (pynative_executor->forward_executor()->enable_async()) {
      auto auto_grad_cell_ptr = grad_executor->top_cell()->auto_grad_cell_ptr();
      auto task = [auto_grad_cell_ptr, context, node, flatten_outputs_value = std::move(flatten_outputs_value),
                   input_tensor_set = std::move(input_tensor_set), input_value_list = std::move(input_value_list),
                   input_value_grad_type = std::move(input_value_grad_type)]() mutable {
        auto_grad_cell_ptr->CallCPPFunctionBprop(flatten_outputs_value, input_tensor_set, context->dirty_inputs_,
                                                 context->non_differentiable_, input_value_list, input_value_grad_type,
                                                 node);
        context->non_differentiable_.clear();
        context->dirty_inputs_.clear();
      };
      grad_executor->DispatchGradQueueTask(std::move(task));
    } else {
      grad_executor->top_cell()->auto_grad_cell_ptr()->CallCPPFunctionBprop(
        flatten_outputs_value, input_tensor_set, context->dirty_inputs_, context->non_differentiable_, input_value_list,
        input_value_grad_type, node);
      context->non_differentiable_.clear();
      context->dirty_inputs_.clear();
    }
  } else {
    MS_LOG(DEBUG) << function_name << " Run in no grad mode";
  }
}

BaseTensorPtrList GradPreProcess(const ValuePtrList &grads, const AbstractBasePtrList &outputs_abstract,
                                 bool materialize_grads, const std::string &function_name) {
  MS_LOG(DEBUG) << function_name << " Begin GradPreProcess";
  BaseTensorPtrList outputs;
  outputs.reserve(grads.size());

  const auto &device_target = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto func_builder = FuncBuilder(function_name, device_target, nullptr);

  for (size_t i = 0; i < grads.size(); ++i) {
    const auto &grad = grads[i];
    BaseTensorPtr tensor = nullptr;
    if (grad->isa<tensor::BaseTensor>()) {
      tensor = grad->cast<BaseTensorPtr>();
    } else if (grad->isa<None>()) {
      if (materialize_grads) {
        const auto filled_zeros_grad = func_builder.FillZeros(grad, outputs_abstract[i]);
        tensor = filled_zeros_grad->cast<BaseTensorPtr>();
      }
    } else {
      MS_LOG(EXCEPTION) << "The value is not a BaseTensor or None.";
    }
    (void)outputs.emplace_back(tensor);
  }
  return outputs;
}

ValuePtrList GradPostProcess(const BaseTensorPtrList &outputs, std::vector<bool> is_tensor_input,
                             const std::string &function_name) {
  MS_LOG(DEBUG) << function_name << " Begin GradPostProcess";
  const auto num_forward_inputs = is_tensor_input.size();
  const auto num_grad_outputs = outputs.size();
  if (num_grad_outputs != num_forward_inputs) {
    std::string msg(function_name);
    msg += " returned an incorrect number of gradients (expected ";
    msg += std::to_string(num_forward_inputs) + ", got ";
    msg += std::to_string(num_grad_outputs) + ")";
    MS_LOG(EXCEPTION) << msg;
  }
  BaseTensorPtrList result;
  result.reserve(num_grad_outputs);
  for (size_t i = 0; i < num_grad_outputs; i++) {
    if (!is_tensor_input[i]) {
      if (outputs[i] != nullptr) {
        std::string msg(function_name);
        msg += " returned a gradient different that is defined at position ";
        msg += std::to_string(i + 1) + ", std the corresponding forward input was not a Tensor";
        MS_LOG(EXCEPTION) << msg;
      }
    } else {
      (void)result.emplace_back(outputs[i]);
    }
  }

  return PyNativeAlgo::DataConvert::TensorListToValueList(result);
}
}  // namespace mindspore::pynative::autograd
