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

#include "kernel/ascend/pyboost/customize/inplace_divmod.h"
#include <memory>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr InplaceDivModAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                   const BaseTensorPtr &input_tensor, const BaseTensorPtr &other_tensor,
                                                   const std::optional<Int64ImmPtr> &rounding_mode) {
  MS_LOG(DEBUG) << "Call InplaceDivModAscendCustomize start";
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, other_tensor);
  op->set_outputs({input_tensor});
  auto mode = rounding_mode.has_value() ? GetValue<int64_t>(rounding_mode.value()) : 0;
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, other_tensor, mode]() {
    MS_LOG(DEBUG) << "Run device task InplaceDivMod start";
    auto device_context = op->device_context();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor, other_tensor);

    LAUNCH_ACLNN(aclnnInplaceDivMod, device_context, op->stream_id(), input_tensor, other_tensor, mode);
    MS_LOG(DEBUG) << "Run device task InplaceDivMod end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
