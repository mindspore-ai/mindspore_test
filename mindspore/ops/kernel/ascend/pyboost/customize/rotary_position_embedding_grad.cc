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
#include "kernel/common/pyboost/auto_generate/copy.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void RotaryPositionEmbeddingGradAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &dy_tensor,
                                                const BaseTensorPtr &cos_tensor, const BaseTensorPtr &sin_tensor,
                                                const std::optional<BaseTensorPtr> &dx_tensor,
                                                const Int64ImmPtr &mode) {
  OpRunner::InferOpOutput(op, dy_tensor, cos_tensor, sin_tensor, dx_tensor, mode);

  // Convert ValuePtr to c++ scalar
  auto mode_imm = GetValue<int64_t>(mode);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), dy_tensor, cos_tensor, sin_tensor, dx_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  BaseTensorPtr dy_tensor_contiguous = dy_tensor;
  BaseTensorPtr cos_tensor_contiguous = cos_tensor;
  BaseTensorPtr sin_tensor_contiguous = sin_tensor;
  if (!dy_tensor->is_contiguous()) {
    MS_LOG(DEBUG) << "For RotaryPositionEmbeddingGrad, dy_tensor is not contiguous.";
    auto copy_op = CREATE_PYBOOST_OP(Copy, kAscendDevice);
    copy_op->set_stream_id(op->stream_id());
    dy_tensor_contiguous = copy_op->Call(dy_tensor);
  }
  if (!cos_tensor->is_contiguous()) {
    MS_LOG(DEBUG) << "For RotaryPositionEmbeddingGrad, cos_tensor is not contiguous.";
    auto copy_op = CREATE_PYBOOST_OP(Copy, kAscendDevice);
    copy_op->set_stream_id(op->stream_id());
    cos_tensor_contiguous = copy_op->Call(cos_tensor);
  }
  if (!sin_tensor->is_contiguous()) {
    MS_LOG(DEBUG) << "For RotaryPositionEmbeddingGrad, sin_tensor is not contiguous.";
    auto copy_op = CREATE_PYBOOST_OP(Copy, kAscendDevice);
    copy_op->set_stream_id(op->stream_id());
    sin_tensor_contiguous = copy_op->Call(sin_tensor);
  }

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, dy_tensor_contiguous, cos_tensor_contiguous, sin_tensor_contiguous, dx_tensor, mode_imm]() {
      MS_LOG(DEBUG) << "Run device task RotaryPositionEmbeddingGrad start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, dy_tensor_contiguous, cos_tensor_contiguous, sin_tensor_contiguous,
                                   dx_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      if (dx_tensor.has_value()) {
        LAUNCH_ACLNN(aclnnRotaryPositionEmbeddingGrad, device_context, op->stream_id(), dy_tensor_contiguous,
                     cos_tensor_contiguous, sin_tensor_contiguous, dx_tensor.value(), mode_imm, outputs[kIndex0],
                     outputs[kIndex1], outputs[kIndex2]);
      } else {
        LAUNCH_ACLNN(aclnnRotaryPositionEmbeddingGrad, device_context, op->stream_id(), dy_tensor_contiguous,
                     cos_tensor_contiguous, sin_tensor_contiguous, nullptr, mode_imm, outputs[kIndex0],
                     outputs[kIndex1], outputs[kIndex2]);
      }

      MS_LOG(DEBUG) << "Run device task RotaryPositionEmbeddingGrad end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
