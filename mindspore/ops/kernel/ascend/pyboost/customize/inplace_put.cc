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

#include "kernel/ascend/pyboost/customize/inplace_put.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr InplacePutAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                                const BaseTensorPtr &index, const BaseTensorPtr &source,
                                                const BoolImmPtr &accumulate) {
  MS_LOG(DEBUG) << "Call aclnnInplacePut start";

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, index, source);
  auto accumulate_imm = GetValue<bool>(accumulate);

  op->set_outputs({input_tensor});
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, index, source, accumulate_imm]() {
      auto device_context = op->device_context();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor, index, source);

      // Inplace output need be front
      LAUNCH_ACLNN(aclnnInplacePut, device_context, op->stream_id(), input_tensor, index, source, accumulate_imm);
      MS_LOG(DEBUG) << "Launch aclnnInplacePut end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
