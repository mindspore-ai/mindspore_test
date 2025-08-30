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

#include "kernel/cpu/native/round_cpu_kernel.h"
#include <algorithm>
#include <memory>

namespace mindspore {
namespace kernel {
namespace round_cpu {

bool RoundCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int RoundCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  int ret = KernelMod::Resize(inputs, outputs);
  return ret;
}

template <typename T>
bool RoundCpuKernelMod::LaunchKernel(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  const auto *input = reinterpret_cast<T *>(inputs[0]->device_ptr());
  auto *output = reinterpret_cast<T *>(outputs[0]->device_ptr());
  const size_t lens = outputs[0]->size() / sizeof(T);

  auto decimals = inputs[1]->GetValueWithCheck<int64_t>();
  if (decimals != 0) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_ << " only support decimals equal 0, but got decimals equal "
                      << decimals;
    return false;
  }

  auto task = [&input, &output](size_t start, size_t end) {
    for (size_t i = start; i < end; i++) {
      output[i] = static_cast<T>(nearbyint(input[i]));
    }
  };
  ParallelLaunchAutoSearch(task, lens, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, RoundCpuKernelMod::RoundFunc>> RoundCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeInt32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt32),
   &RoundCpuKernelMod::LaunchKernel<int>},
  {KernelAttr().AddInputAttr(kNumberTypeInt64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64),
   &RoundCpuKernelMod::LaunchKernel<int64_t>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
   &RoundCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
   &RoundCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> RoundCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, RoundFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Round, RoundCpuKernelMod);
}  // namespace round_cpu
}  // namespace kernel
}  // namespace mindspore
