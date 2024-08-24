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

#ifndef MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_CHECK_OVERFLOW_H_
#define MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_CHECK_OVERFLOW_H_
#include <set>
#include <string>
#include <vector>
#include "debug/data_dump/device_statistic/statistic_kernel.h"
#include "op_def/nn_op_name.h"

namespace mindspore {

namespace datadump {

inline const std::set<TypeId> overflow_supported_dtype{kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};

class CheckOverflowKernel : public StatisticKernel {
 public:
  explicit CheckOverflowKernel(const DeviceContext *device_context, std::uint32_t stream_id)
      : StatisticKernel(device_context, kAllFiniteOpName, overflow_supported_dtype, stream_id) {}

  vector<KernelTensor *> CheckInputs(vector<KernelTensor *> inputs);
  DeviceAddressPtr LaunchKernelAsync(KernelTensor *input) = delete;
  DeviceAddressPtr LaunchKernelAsync(vector<KernelTensor *> inputs) override;
};

}  // namespace datadump
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_DEBUG_DEVICE_STATISTIC_CHECK_OVERFLOW_H_
