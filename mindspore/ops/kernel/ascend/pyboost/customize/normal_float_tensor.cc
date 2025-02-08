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

#include "kernel/ascend/pyboost/customize/normal_float_tensor.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr NormalFloatTensorAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                       const FP32ImmPtr &mean_float, const BaseTensorPtr &std_tensor,
                                                       const BaseTensorPtr &seed, const BaseTensorPtr &offset) {
  MS_LOG(DEBUG) << "NormalFloatTensor call start";
  OpRunner::InferOpOutput(op, mean_float, std_tensor, seed, offset);
  // ValueTuple to std::vector
  auto mean_imm = GetValue<float>(mean_float);
  auto [seed_imm, offset_imm] = UpdateGeneratorState(seed, offset);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), std_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, mean_imm, std_tensor, seed_imm, offset_imm]() {
      MS_LOG(DEBUG) << "Run device task NormalFloatTensor end";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, std_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnNormalFloatTensor, device_context, op->stream_id(), mean_imm, std_tensor, seed_imm, offset_imm,
                   outputs[0]);
      MS_LOG(DEBUG) << "Run device task NormalFloatTensor end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
