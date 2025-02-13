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

#include "kernel/ascend/pyboost/customize/soft_margin_loss_grad.h"
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
tensor::BaseTensorPtr SoftMarginLossGradAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                        const BaseTensorPtr &predict_tensor,
                                                        const BaseTensorPtr &label_tensor,
                                                        const BaseTensorPtr &dout_tensor,
                                                        const Int64ImmPtr &reduction) {
  MS_LOG(DEBUG) << "SoftMarginLossGrad call start";
  OpRunner::InferOpOutput(op, predict_tensor, label_tensor, dout_tensor, reduction);
  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  auto reduction_value = ops::ConvertReductionForAclnn(reduction_imm);

  // No need to convert input
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), predict_tensor, label_tensor, dout_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, dout_tensor, predict_tensor, label_tensor, reduction_value]() {
      MS_LOG(DEBUG) << "Run device task SoftMarginLossGrad end";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, dout_tensor, predict_tensor, label_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnSoftMarginLossBackward, device_context, op->stream_id(), dout_tensor, predict_tensor,
                   label_tensor, reduction_value, outputs[0]);
      MS_LOG(DEBUG) << "Run device task SoftMarginLossGrad end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
