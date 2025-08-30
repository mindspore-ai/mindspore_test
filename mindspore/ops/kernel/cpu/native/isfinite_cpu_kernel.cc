/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/native/isfinite_cpu_kernel.h"
#include <cmath>

namespace mindspore {
namespace kernel {
namespace isfinite_cpu {
namespace {
constexpr size_t kIsFiniteInputsNum = 1;
constexpr size_t kIsFiniteOutputsNum = 1;
}  // namespace

bool IsFiniteCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIsFiniteInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIsFiniteOutputsNum, kernel_name_);
  if (!MatchKernelFunc(kernel_name_, inputs, outputs)) {
    return false;
  }
  return true;
}

int IsFiniteCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return KRET_OK;
}

template <typename T>
bool IsFiniteCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                                        const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kIsFiniteInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kIsFiniteOutputsNum, kernel_name_);
  auto input = GetDeviceAddress<T>(inputs, kIndex0);
  auto output = GetDeviceAddress<bool>(outputs, kIndex0);

  CTask task;
  task = [this, &input, &output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      if ((std::is_same_v<T, float16>) || (std::is_same_v<T, float>)) {
        float temp_num = static_cast<float>(input[i]);
        output[i] = !std::isinf(temp_num) && !std::isnan(temp_num);
      } else if (std::is_same_v<T, double>) {
        double temp_num = static_cast<double>(input[i]);
        output[i] = !std::isinf(temp_num) && !std::isnan(temp_num);
      } else {
        output[i] = true;
      }
    }
  };
  size_t elem_num = outputs[kIndex0]->size() / sizeof(bool);
  ParallelLaunch(task, elem_num, 0, this, pool_);
  return true;
}

const std::vector<std::pair<KernelAttr, IsFiniteCpuKernelMod::KernelRunFunc>> IsFiniteCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeBool).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<bool>},
  {KernelAttr().AddInputAttr(kNumberTypeInt8).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<int8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt16).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<int16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<int32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<float16>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<double>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt8).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<uint8_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt16).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<uint16_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt32).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<uint32_t>},
  {KernelAttr().AddInputAttr(kNumberTypeUInt64).AddOutputAttr(kNumberTypeBool),
   &IsFiniteCpuKernelMod::LaunchKernel<uint64_t>},
};

const std::vector<std::pair<KernelAttr, IsFiniteCpuKernelMod::KernelRunFunc>> &IsFiniteCpuKernelMod::GetFuncList()
  const {
  return func_list_;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, IsFinite, IsFiniteCpuKernelMod);
}  // namespace isfinite_cpu
}  // namespace kernel
}  // namespace mindspore
