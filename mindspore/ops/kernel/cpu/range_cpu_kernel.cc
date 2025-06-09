/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/range_cpu_kernel.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace range_cpu {
namespace {
constexpr size_t kRangeInputsNum = 4;
constexpr size_t kRangeOutputsNum = 1;

template <typename T>
T Sign(T num) {
  if (num > static_cast<T>(0.0)) {
    return static_cast<T>(1.0);
  } else if (num == static_cast<T>(0.0)) {
    return static_cast<T>(0.0);
  } else {
    return static_cast<T>(-1.0);
  }
}
}  // namespace

bool RangeCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kRangeInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kRangeOutputsNum, kernel_name_);
  return MatchKernelFunc(kernel_name_, inputs, outputs);
}

template <typename S, typename T>
bool RangeCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                     const std::vector<KernelTensor *> &outputs) {
  auto start = GetDeviceAddress<S>(inputs, kIndex0)[kIndex0];
  auto limit = GetDeviceAddress<S>(inputs, kIndex1)[kIndex0];
  auto delta = GetDeviceAddress<S>(inputs, kIndex2)[kIndex0];
  if (delta == static_cast<S>(0)) {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", the delta can not be 0.";
    return false;
  }

  auto output = GetDeviceAddress<T>(outputs, kIndex0);
  size_t output_size = outputs[kIndex0]->size() / sizeof(T);
  if (Sign(delta) * Sign(limit - start) >= 0) {
    for (int index = 0; index < SizeToInt(output_size); index++) {
      output[index] = static_cast<T>(delta * index + start);
    }
  } else {
    MS_LOG(ERROR) << "For " << kernel_name_ << ", upper bound and larger bound inconsistent with step sign.";
    return false;
  }
  return true;
}

const std::vector<std::pair<KernelAttr, RangeCpuKernelMod::KernelRunFunc>> &RangeCpuKernelMod::GetFuncList() const {
  static const std::vector<std::pair<KernelAttr, RangeCpuKernelMod::KernelRunFunc>> func_list = {
    {KernelAttr()
       .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &RangeCpuKernelMod::LaunchKernel<float, float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeFloat64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeFloat32),
     &RangeCpuKernelMod::LaunchKernel<double, float>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt32)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt32),
     &RangeCpuKernelMod::LaunchKernel<int32_t, int32_t>},
    {KernelAttr()
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
       .AddOutputAttr(kNumberTypeInt64),
     &RangeCpuKernelMod::LaunchKernel<int64_t, int64_t>}};
  return func_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Range, RangeCpuKernelMod);
}  // namespace range_cpu
}  // namespace kernel
}  // namespace mindspore
