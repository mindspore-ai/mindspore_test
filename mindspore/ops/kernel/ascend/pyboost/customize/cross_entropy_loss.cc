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

#include "kernel/ascend/pyboost/customize/cross_entropy_loss.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <tuple>
#include "include/common/utils/utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> CrossEntropyLossAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &input, const TensorPtr &target,
  const std::optional<TensorPtr> &weight, const Int64ImmPtr &reduction, const Int64ImmPtr &ignore_index,
  const FP64ImmPtr &label_smoothing, const FP64ImmPtr &lse_square_scale_for_zloss, const BoolImmPtr &return_zloss) {
  OpRunner::InferOpOutput(op, input, target, weight, reduction, ignore_index, label_smoothing,
                          lse_square_scale_for_zloss, return_zloss);
  auto reduction_imm = static_cast<Reduction>(GetValue<int64_t>(reduction));
  // transform reduction enum value to corresponding value
  const char *reduction_value = ops::ConvertReductionStrForAclnn(reduction_imm);
  const auto ignore_index_imm = GetValue<int64_t>(ignore_index);
  const auto label_smoothing_imm = static_cast<double>(label_smoothing->value());
  const auto lse_square_scale_for_zloss_imm = static_cast<double>(lse_square_scale_for_zloss->value());
  const auto return_zloss_imm = GetValue<bool>(return_zloss);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input, target, weight);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, input, target, weight, reduction_value, ignore_index_imm, label_smoothing_imm, lse_square_scale_for_zloss_imm,
     return_zloss_imm]() {
      MS_LOG(DEBUG) << "Run device task CrossEntropyLoss start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), input, target, weight);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
      LAUNCH_ACLNN(aclnnCrossEntropyLoss, device_context, op->stream_id(), input, target, weight, reduction_value,
                   ignore_index_imm, label_smoothing_imm, lse_square_scale_for_zloss_imm, return_zloss_imm, outputs[0],
                   outputs[1], outputs[2], outputs[3]);
      MS_LOG(DEBUG) << "Run device task CrossEntropyLoss end";
    }));
  return std::make_tuple(op->output(0), op->output(1), op->output(2), op->output(3));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
