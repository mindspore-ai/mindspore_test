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

#include "kernel/ascend/pyboost/customize/adamw.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<tensor::TensorPtr, tensor::TensorPtr, tensor::TensorPtr> AdamWAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const TensorPtr &var, const TensorPtr &m, const TensorPtr &v,
  const TensorPtr &max_v, const TensorPtr &grad, const TensorPtr &step, const FP32ImmPtr &lr, const FP32ImmPtr &beta1,
  const FP32ImmPtr &beta2, const FP32ImmPtr &decay, const FP32ImmPtr &epsilon, const BoolImmPtr &amsgrad,
  const BoolImmPtr &maximize) {
  const auto lr_imm = GetValue<float>(lr);
  const auto beta1_imm = GetValue<float>(beta1);
  const auto beta2_imm = GetValue<float>(beta2);
  const auto decay_imm = GetValue<float>(decay);
  const auto epsilon_imm = GetValue<float>(epsilon);
  const auto amsgrad_imm = GetValue<bool>(amsgrad);
  const auto maximize_imm = GetValue<bool>(maximize);
  op->set_outputs({var, m, v});

  if (amsgrad_imm) {
    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), var, m, v, max_v, grad, step);
  } else {
    PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), var, m, v, grad, step);
  }

  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, var, m, v, max_v, grad, step, lr_imm, beta1_imm, beta2_imm,
                                                  decay_imm, epsilon_imm, amsgrad_imm, maximize_imm]() {
      auto device_context = op->device_context();

      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      if (amsgrad_imm) {
        PyBoostUtils::MallocOpInputs(device_context, var, m, v, max_v, grad, step);
        LAUNCH_ACLNN(aclnnApplyAdamWV2, device_context, op->stream_id(), var, m, v, max_v, grad, step, lr_imm,
                     beta1_imm, beta2_imm, decay_imm, epsilon_imm, amsgrad_imm, maximize_imm);
      } else {
        PyBoostUtils::MallocOpInputs(device_context, var, m, v, grad, step);
        LAUNCH_ACLNN(aclnnApplyAdamWV2, device_context, op->stream_id(), var, m, v, nullptr, grad, step, lr_imm,
                     beta1_imm, beta2_imm, decay_imm, epsilon_imm, amsgrad_imm, maximize_imm);
      }

      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return std::make_tuple(var, m, v);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
