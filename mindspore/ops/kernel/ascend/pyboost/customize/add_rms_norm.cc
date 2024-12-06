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

#include "kernel/ascend/pyboost/customize/add_rms_norm.h"
#include "op_def/auto_generate/gen_ops_primitive.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::BaseTensorPtr, tensor::BaseTensorPtr, tensor::BaseTensorPtr> AddRmsNormAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x1_tensor, const BaseTensorPtr &x2_tensor,
  const BaseTensorPtr &gamma_tensor, const FP32ImmPtr &epsilon) {
  OpRunner::InferOpOutput(op, x1_tensor, x2_tensor, gamma_tensor, epsilon);
  auto epsilon_imm = static_cast<double>(GetValue<float>(epsilon));

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x1_tensor, x2_tensor, gamma_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, x1_tensor, x2_tensor, gamma_tensor, epsilon_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x1_tensor, x2_tensor, gamma_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnAddRmsNorm, device_context, op->stream_id(), x1_tensor, x2_tensor, gamma_tensor, epsilon_imm,
                   outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call end";
    }));
  return std::make_tuple(op->output(kIndex0), op->output(kIndex1), op->output(kIndex2));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
