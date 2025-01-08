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
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "kernel/ascend/pyboost/customize/triangular_solve.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void TriangularSolveAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &b_tensor,
                                    const BaseTensorPtr &a_tensor, const BoolImmPtr &upper, const BoolImmPtr &transpose,
                                    const BoolImmPtr &unitriangular) {
  OpRunner::InferOpOutput(op, b_tensor, a_tensor, upper, transpose, unitriangular);
  const auto upper_imm = GetValue<bool>(upper);
  const auto transpose_imm = GetValue<bool>(transpose);
  const auto unitriangular_imm = GetValue<bool>(unitriangular);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), b_tensor, a_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, b_tensor, a_tensor, upper_imm, transpose_imm, unitriangular_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, b_tensor, a_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      MS_LOG(DEBUG) << "Run device task op, b_tensor, a_tensor, upper, transpose, unitriangular start";
      LAUNCH_ACLNN(aclnnTriangularSolve, device_context, op->stream_id(), b_tensor, a_tensor, upper_imm, transpose_imm,
                   unitriangular_imm, outputs[kIndex0], outputs[kIndex1]);
      MS_LOG(DEBUG) << "Run device task op, b_tensor, a_tensor, upper, transpose, unitriangular end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
