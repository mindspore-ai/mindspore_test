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
#include "debug/data_dump/device_statistic/kernel_launcher.h"
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "debug/data_dump/device_statistic/common.h"

namespace mindspore {
namespace datadump {

TensorPtr CalStatistic(const std::string &stat_name, const DeviceContext *device_context, KernelTensor *input,
                       const std::uint32_t stream_id) {
  auto out = CalStatisticAsync(stat_name, device_context, input, stream_id);
  return SyncDeviceToHostTensor(out.back());
}

vector<DeviceAddressPtr> CalStatisticAsync(const std::string &stat_name, const DeviceContext *device_context,
                                           KernelTensor *input, const uint32_t stream_id) {
  MS_EXCEPTION_IF_NULL(input);
  auto dtype = input->dtype_id();
  auto kernel = KernelFactory::Instance().CreateKernel(stat_name, device_context);
  if (kernel->CheckDataType(dtype)) {
    return kernel->LaunchKernelAsync(input, stream_id);
  } else {
    const auto &device_name = device_context->device_context_key_.device_name_;
    const auto &type_name = TypeIdToString(dtype);
    WarningOnce(device_name, type_name, stat_name);
    return {nullptr};
  }
}

}  // namespace datadump
}  // namespace mindspore
