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

#include <memory>
#include <string>
#include "kernel/ascend/pyboost/customize/linalg_qr.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ops/op_def/op_enum.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void LinalgQrAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &A_tensor,
                             const Int64ImmPtr &mode) {
  OpRunner::InferOpOutput(op, A_tensor, mode);
  // Convert ValuePtr to c++ scalar
  int64_t mode_imm = GetValue<int64_t>(mode);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), A_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, A_tensor, mode_imm]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, A_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    MS_LOG(DEBUG) << "Run device task aclnnLinalgQr start";
    LAUNCH_ACLNN(aclnnLinalgQr, device_context, op->stream_id(), A_tensor, mode_imm, outputs[kIndex0],
                 outputs[kIndex1]);
    MS_LOG(DEBUG) << "Run device task aclnnLinalgQr end";
  }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
