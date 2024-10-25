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

#include "kernel/ascend/pyboost/customize/rotary_position_embedding_grad.h"
#include <string>
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "op_def/auto_generate/gen_ops_primitive.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void RotaryPositionEmbeddingGradAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &dy_tensor,
                                                const BaseTensorPtr &cos_tensor, const BaseTensorPtr &sin_tensor,
                                                const std::optional<BaseTensorPtr> &dx_tensor,
                                                const Int64ImmPtr &mode) {
  OpRunner::InferOpOutput(op, dy_tensor, cos_tensor, sin_tensor, dx_tensor, mode);
  // ValueTuple to std::vector

  // Convert ValuePtr to c++ scalar
  // Convert ValuePtr to c++ scalar
  auto mode_imm = GetValue<int64_t>(mode);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), dy_tensor, cos_tensor, sin_tensor, dx_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, dy_tensor, cos_tensor, sin_tensor, dx_tensor, mode_imm]() {
      MS_LOG(DEBUG) << "Run device task RotaryPositionEmbeddingGrad start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, dy_tensor, cos_tensor, sin_tensor, dx_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      if (dx_tensor.has_value()) {
        LAUNCH_ACLNN(aclnnRotaryPositionEmbeddingGrad, device_context, op->stream_id(), dy_tensor, cos_tensor,
                     sin_tensor, dx_tensor.value(), mode_imm, outputs[0], outputs[1], outputs[2]);
      } else {
        LAUNCH_ACLNN(aclnnRotaryPositionEmbeddingGrad, device_context, op->stream_id(), dy_tensor, cos_tensor,
                     sin_tensor, nullptr, mode_imm, outputs[0], outputs[1], outputs[2]);
      }

      MS_LOG(DEBUG) << "Run device task RotaryPositionEmbeddingGrad end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
