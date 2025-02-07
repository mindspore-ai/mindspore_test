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

#include "kernel/ascend/pyboost/customize/eye.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr EyeAscendCustomize(const std::shared_ptr<OpRunner> &op, const Int64ImmPtr &n,
                                         const Int64ImmPtr &m, const Int64ImmPtr &dtype) {
  OpRunner::InferOpOutput(op, n, m, dtype);

  auto n_imm = GetValue<int64_t>(n);
  auto m_imm = GetValue<int64_t>(m);

  // No need to convert input
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, n_imm, m_imm]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    MS_LOG(DEBUG) << "Call Eye start";
    LAUNCH_ACLNN(aclnnEye, device_context, op->stream_id(), n_imm, m_imm, outputs[0]);
    MS_LOG(DEBUG) << "Launch Eye end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
