/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "kernel/cpu/hypot_cpu_kernel.h"

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <algorithm>
#include <utility>
#include <numeric>

#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace hypot_cpu {
namespace {
const size_t kHypotInputsNum = 2;
const size_t kHypotOutputsNum = 1;
}  // namespace

bool HypotCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kHypotInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kHypotOutputsNum, kernel_name_);
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << kernel_name_ << " does not support this kernel data type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  return true;
}

int HypotCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  x1_shape_ = inputs[0]->GetShapeVector();
  x2_shape_ = inputs[1]->GetShapeVector();
  y_shape_ = outputs[0]->GetShapeVector();
  return KRET_OK;
}

template <typename T>
bool HypotCpuKernelMod::LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                                     const std::vector<kernel::KernelTensor *> &outputs) {
  const T *x1 = GetDeviceAddress<const T>(inputs, 0);
  const T *x2 = GetDeviceAddress<const T>(inputs, 1);
  T *y = GetDeviceAddress<T>(outputs, 0);
  if (y_shape_.size() == 0) {
    (void)y_shape_.insert(y_shape_.begin(), 1);
  }
  auto output_size = SizeOf(y_shape_);

  BroadcastIterator base_iter(x1_shape_, x2_shape_, y_shape_);
  auto task = [this, &x1, &x2, &y, &base_iter](size_t start, size_t end) {
    auto iter = base_iter;
    iter.SetPos(start);
    for (size_t i = start; i < end; i++) {
      y[i] = std::hypot(x1[iter.GetInputPosA()], x2[iter.GetInputPosB()]);
      iter.GenNextPos();
    }
  };
  ParallelLaunchAutoSearch(task, output_size, this, &parallel_search_info_);
  return true;
}

std::vector<std::pair<KernelAttr, HypotCpuKernelMod::HypotLaunchFunc>> HypotCpuKernelMod::func_list_ = {
  {KernelAttr().AddInputAttr(kNumberTypeFloat32).AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
   &HypotCpuKernelMod::LaunchKernel<float>},
  {KernelAttr().AddInputAttr(kNumberTypeFloat64).AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
   &HypotCpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> HypotCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, HypotLaunchFunc> &pair) { return pair.first; });

  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Hypot, HypotCpuKernelMod);
}  // namespace hypot_cpu
}  // namespace kernel
}  // namespace mindspore
