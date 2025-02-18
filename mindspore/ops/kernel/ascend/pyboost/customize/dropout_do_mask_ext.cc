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

#include "kernel/ascend/pyboost/customize/dropout_do_mask_ext.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr DropoutDoMaskExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input,
                                                      const BaseTensorPtr &mask, const FP32ImmPtr &p) {
  OpRunner::InferOpOutput(op, input, mask, p);
  // Create device address for input/output tensors.
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input, mask);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input, mask, p]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(op->device_context(), input, mask);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
    auto p_value = static_cast<double>(p->value());

    LAUNCH_ACLNN(aclnnDropoutDoMask, device_context, op->stream_id(), input, mask, p_value, outputs[kIndex0]);
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
