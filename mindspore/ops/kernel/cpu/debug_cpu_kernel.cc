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

#include "kernel/cpu/debug_cpu_kernel.h"
#include "plugin/res_manager/cpu/cpu_device_address/cpu_device_address.h"

namespace mindspore {
namespace kernel {
namespace debug_cpu {
namespace {
constexpr size_t kDebugInputsNum = 1;
constexpr size_t kDebugOutputsNum = 1;
}  // namespace

bool DebugCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  return true;
}

bool DebugCpuKernelMod::Launch(const std::vector<kernel::KernelTensor *> &inputs,
                               const std::vector<kernel::KernelTensor *> &,
                               const std::vector<kernel::KernelTensor *> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kDebugInputsNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kDebugOutputsNum, kernel_name_);
  const auto *val = GetDeviceAddress<int>(inputs, kIndex0);
  MS_LOG(DEBUG) << " launch DebugCpuKernelMod";

  auto *output = GetDeviceAddress<int>(outputs, kIndex0);
  size_t elem_num = inputs[0]->size() / sizeof(int);
  for (size_t i = 0; i < elem_num; i++) {
    output[i] = static_cast<int>(val[i]);
  }
  return true;
}

std::vector<KernelAttr> DebugCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeInt32)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, Debug, DebugCpuKernelMod);
}  // namespace debug_cpu
}  // namespace kernel
}  // namespace mindspore
