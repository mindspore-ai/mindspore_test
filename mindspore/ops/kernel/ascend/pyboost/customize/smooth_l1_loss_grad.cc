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

#include "kernel/ascend/pyboost/customize/smooth_l1_loss_grad.h"
#include <memory>
#include <unordered_map>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindapi/base/types.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr SmoothL1LossGradAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                  const TensorPtr &prediction_tensor, const TensorPtr &target_tensor,
                                                  const TensorPtr &dout_tensor, const FP64ImmPtr &beta,
                                                  const Int64ImmPtr &reduction) {
  MS_LOG(DEBUG) << "SmoothL1LossGrad call start";
  OpRunner::InferOpOutput(op, prediction_tensor, target_tensor, dout_tensor, beta, reduction);
  // Convert ValuePtr to c++ scalar
  auto beta_imm = static_cast<float>(beta->value());
  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  auto reduction_value = ops::ConvertReductionForAclnn(reduction_imm);

  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), prediction_tensor, target_tensor, dout_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, dout_tensor, prediction_tensor, target_tensor, beta_imm, reduction_value]() {
      MS_LOG(DEBUG) << "Run device task SmoothL1LossGrad end";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, dout_tensor, prediction_tensor, target_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnSmoothL1LossBackward, device_context, op->stream_id(), dout_tensor, prediction_tensor,
                   target_tensor, reduction_value, beta_imm, outputs[0]);
      MS_LOG(DEBUG) << "Run device task SmoothL1LossGrad end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
