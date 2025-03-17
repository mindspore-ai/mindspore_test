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

#include <string>
#include <memory>
#include "kernel/ascend/pyboost/customize/angle_ext.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void AngleExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor) {
  OpRunner::InferOpOutput(op, input_tensor);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor]() {
    MS_LOG(DEBUG) << "Run device task AngleExt start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnAngleV2, device_context, op->stream_id(), input_tensor, outputs[kIndex0]);
    MS_LOG(DEBUG) << "Run device task AngleExt end";
  }));

  return;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
