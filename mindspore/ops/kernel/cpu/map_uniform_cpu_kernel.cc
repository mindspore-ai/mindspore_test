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

#include "kernel/cpu/map_uniform_cpu_kernel.h"
#include <string>
#include <memory>
#include <vector>
#include <algorithm>
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kMapUniformInputsNum = 3;
constexpr size_t kMapUniformOutputsNum = 1;
}  // namespace

bool MapUniformCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kMapUniformInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kMapUniformOutputsNum, kernel_name_);
  dtype_ = inputs[kIndex0]->dtype_id();

  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int MapUniformCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  if (int ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }

  auto input_x_shape = inputs[kIndex0]->GetShapeVector();
  batch_size_ = SizeOf(input_x_shape);
  return KRET_OK;
}

template <typename T>
bool MapUniformCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                          const std::vector<kernel::KernelTensor *> &,
                                          const std::vector<kernel::KernelTensor *> &outputs) {
  MS_LOG(INFO) << "Input size: " << batch_size_;
  auto input_x = GetDeviceAddress<T>(inputs, kIndex0);
  auto per_group_size = *GetDeviceAddress<T>(inputs, kIndex1);
  auto group_num = *GetDeviceAddress<T>(inputs, kIndex2);
  auto output_x = GetDeviceAddress<T>(outputs, kIndex0);
  if (group_num <= 0) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', the 'group_num' must be greater than 0, but got " << group_num;
  }
  T max_num = group_num * per_group_size;
  for (size_t i = 0; i < batch_size_; ++i) {
    output_x[i] = input_x[i] % group_num * per_group_size + input_x[i] / group_num;
    if (output_x[i] >= max_num) {
      MS_LOG(EXCEPTION) << "For '" << kernel_name_ << "', all elements in output must be less than " << max_num
                        << ", but got " << output_x[i];
    }
  }
  return true;
}

std::vector<std::pair<KernelAttr, MapUniformCpuKernelMod::MapUniformFunc>> MapUniformCpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddInputAttr(kNumberTypeInt32)
     .AddOutputAttr(kNumberTypeInt32),
   &MapUniformCpuKernelMod::LaunchKernel<int>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddInputAttr(kNumberTypeInt64)
     .AddOutputAttr(kNumberTypeInt64),
   &MapUniformCpuKernelMod::LaunchKernel<int64_t>}};

std::vector<KernelAttr> MapUniformCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, MapUniformFunc> &item) { return item.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, MapUniform, MapUniformCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
