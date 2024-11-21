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

#include "kernel/ascend/pyboost/customize/divmod.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/op_register.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr DivModAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor,
                                            const BaseTensorPtr &y_tensor,
                                            const std::optional<Int64ImmPtr> &rounding_mode) {
  OpRunner::InferOpOutput(op, x_tensor, y_tensor, rounding_mode);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor, y_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, x_tensor, y_tensor, rounding_mode]() {
    MS_LOG(DEBUG) << "Run device task DivMod start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, x_tensor, y_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    auto mode = 0;
    if (rounding_mode.has_value()) {
      mode = GetValue<int64_t>(rounding_mode.value());
    }

    LAUNCH_ACLNN(aclnnDivMod, device_context, op->stream_id(), x_tensor, y_tensor, mode, outputs[0]);
    MS_LOG(DEBUG) << "Run device task DivMod end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
