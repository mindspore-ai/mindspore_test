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

#include "kernel/ascend/pyboost/customize/quant_grouped_matmul_dequant.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include "include/common/utils/utils.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void QuantGroupedMatmulDequantAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x_tensor, const BaseTensorPtr &weight_tensor,
  const BaseTensorPtr &weight_scale_tensor, const BaseTensorPtr &group_list_tensor,
  const std::optional<BaseTensorPtr> &bias_tensor, const std::optional<BaseTensorPtr> &x_scale_tensor,
  const std::optional<BaseTensorPtr> &x_offset_tensor, const std::optional<BaseTensorPtr> &smmoth_scale_tensor,
  const mindspore::StringImmPtr &x_quant_mode, const mindspore::BoolImmPtr &transpose_weight) {
  OpRunner::InferOpOutput(op, x_tensor, weight_tensor, weight_scale_tensor, group_list_tensor, bias_tensor,
                          x_scale_tensor, x_offset_tensor, smmoth_scale_tensor, x_quant_mode, transpose_weight);

  std::string x_quant_mode_imm = GetValue<std::string>(x_quant_mode);
  bool transpose_weight_imm = GetValue<bool>(transpose_weight);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), weight_tensor, x_tensor, weight_scale_tensor,
                                group_list_tensor, bias_tensor, x_scale_tensor, x_offset_tensor, smmoth_scale_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x_tensor, weight_tensor, weight_scale_tensor, group_list_tensor, bias_tensor, x_scale_tensor, x_offset_tensor,
     smmoth_scale_tensor, x_quant_mode_imm, transpose_weight_imm]() {
      MS_LOG(DEBUG) << "Run device task QuantGroupedMatmulDequant start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), x_tensor, weight_tensor, weight_scale_tensor,
                                   group_list_tensor, bias_tensor, x_scale_tensor, x_offset_tensor,
                                   smmoth_scale_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
      LAUNCH_ACLNN(aclnnQuantGroupedMatmulDequant, device_context, op->stream_id(), x_tensor, weight_tensor,
                   weight_scale_tensor, group_list_tensor, bias_tensor, x_scale_tensor, x_offset_tensor,
                   smmoth_scale_tensor, x_quant_mode_imm, transpose_weight_imm, outputs[0]);
      MS_LOG(DEBUG) << "Run device task QuantGroupedMatmulDequant end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
