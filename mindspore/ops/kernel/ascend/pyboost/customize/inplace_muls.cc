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

#include "kernel/ascend/pyboost/customize/inplace_muls.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/op_register.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "kernel/ascend/pyboost/customize/inplace_mul.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
static inline bool CanCast(const ScalarPtr &from, const BaseTensorPtr &to) {
  if (PyBoostUtils::IsFloat(from) && PyBoostUtils::IsIntegral(to)) {
    return false;
  }
  if (!PyBoostUtils::IsBool(from) && PyBoostUtils::IsBool(to)) {
    return false;
  }
  return true;
}

tensor::BaseTensorPtr InplaceMulsAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_tensor,
                                                 const ScalarPtr &other) {
  MS_LOG(DEBUG) << "Call InplaceMuls start";
  // Align Pytorch's logic on arithmetic operations.
  // For details, please refer to "torch.dtype".
  if (MS_UNLIKELY(!CanCast(other, input_tensor))) {
    MS_EXCEPTION(TypeError) << "For " << op->primitive()->name() << ", other type " << other->type()
                            << " can't be cast to the desired output type " << input_tensor->Dtype();
  }
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  op->set_outputs({input_tensor});

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, other]() {
    MS_LOG(DEBUG) << "Run device task InplaceMuls start";
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);

    // Inplace output need be front
    LAUNCH_ACLNN(aclnnInplaceMuls, device_context, op->stream_id(), input_tensor, other);
    MS_LOG(DEBUG) << "Launch InplaceMuls end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
