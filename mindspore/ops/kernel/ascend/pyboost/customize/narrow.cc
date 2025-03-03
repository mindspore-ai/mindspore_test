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
#include "kernel/ascend/pyboost/customize/narrow.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr NarrowAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                        const Int64ImmPtr &dim, const Int64ImmPtr &start, const Int64ImmPtr &length) {
  OpRunner::InferOpOutput(op, input_tensor, dim, start, length);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  const auto &input_shape = input_tensor->shape();
  auto dim_value = dim->value();
  dim_value = dim_value < 0 ? dim_value + SizeToLong(input_shape.size()) : dim_value;
  auto start_value = start->value();
  start_value = start_value < 0 ? start_value + input_shape.at(dim_value) : start_value;
  auto end_value = start_value + length->value();
  const int64_t step_value = 1;

  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, dim_value, start_value, end_value, step_value]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), input_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
      // Launch aclnn
      if (start_value != end_value) {
        LAUNCH_ACLNN(aclnnSlice, device_context, op->stream_id(), input_tensor, dim_value, start_value, end_value,
                     step_value, outputs[0]);
      }
    }));

  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
