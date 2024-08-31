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

#include "kernel/cpu/pyboost/customize/select.h"
#include "kernel/cpu/pyboost/auto_generate/cast.h"
#include "kernel/cpu/pyboost/auto_generate/select.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/common/pyboost/op_runner.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void SelectCPUCall(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &condition_tensor,
                   const BaseTensorPtr &x_tensor, const BaseTensorPtr &y_tensor,
                   const std::vector<AbstractBasePtr> &input_abs) {
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), condition_tensor, x_tensor, y_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, condition_tensor, x_tensor, y_tensor, input_abs]() {
      MS_LOG(DEBUG) << "For 'Select', the cpu task 'Select' start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      const auto primitive = std::make_shared<Primitive>(prim::kPrimSelect->name());
      MS_EXCEPTION_IF_NULL(primitive);

      PyBoostUtils::MallocOpInputs(device_context, condition_tensor, x_tensor, y_tensor);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      const auto &input_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), input_abs, condition_tensor, x_tensor, y_tensor);
      const auto &output_address_info =
        PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->output_abs()}, outputs);

      PyBoostUtils::LaunchKernel(primitive, device_context, input_address_info, output_address_info);
      MS_LOG(DEBUG) << "For 'Select', the cpu task 'Select' end";
    }));
}
}  // namespace

void SelectCPUCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &condition_tensor,
                        const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor) {
  OpRunner::InferOpOutput(op, condition_tensor, input_tensor, other_tensor);

  // Infer function has confirmed the actual dtype of output
  TypeId out_dtype = op->output_value_simple_info()->dtype_vector_[kIndex0]->type_id();

  // Call Cast before Launch Select
  BaseTensorPtr x_tensor = input_tensor;
  if (input_tensor->data_type() != out_dtype) {
    MS_LOG(DEBUG) << "Call Cast cpu kernel, src dtype: " << TypeIdToString(input_tensor->data_type())
                  << ", dst dtype: " << TypeIdToString(out_dtype);
    x_tensor =
      PyBoostUtils::CastTensor(input_tensor, out_dtype, op->device_context()->device_context_key_.device_name_);
  }

  BaseTensorPtr y_tensor = other_tensor;
  if (other_tensor->data_type() != out_dtype) {
    MS_LOG(DEBUG) << "Call Cast cpu kernel, src dtype: " << TypeIdToString(other_tensor->data_type())
                  << ", dst dtype: " << TypeIdToString(out_dtype);
    y_tensor =
      PyBoostUtils::CastTensor(other_tensor, out_dtype, op->device_context()->device_context_key_.device_name_);
  }

  // Set new input abstract for Select
  std::vector<AbstractBasePtr> new_input_abs{condition_tensor->ToAbstract(), x_tensor->ToAbstract(),
                                             y_tensor->ToAbstract()};
  SelectCPUCall(op, condition_tensor, x_tensor, y_tensor, new_input_abs);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
