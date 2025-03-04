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
 * limitations under the License.plugin/device/cpu/hal/device
 */

#include "mindspore/ops/kernel/cpu/pyboost/customize/inplace_copy.h"
#include <memory>
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void InplaceCopyCPUCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &variable,
                             const BaseTensorPtr &value) {
  MS_LOG(DEBUG) << "InplaceCopy cpu pyboost call start";
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), variable, value);
  // Set inplace output
  op->set_outputs({variable});
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, variable, value]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, variable, value);

    const auto &input_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), op->input_abs(), variable, value);
    const auto &output_address_info =
      PyBoostUtils::GetAddressInfo(device_context, op->stream_id(), {op->input_abs()}, outputs);

    PyBoostUtils::LaunchKernel(op->primitive(), op->device_context(), input_address_info, output_address_info);
    MS_LOG(DEBUG) << "InplaceCopy cpu pyboost launch end";
  }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
