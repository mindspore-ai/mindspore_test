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
 * limitations under the License.
 */

#include "kernel/cpu/native/sequence/bool_binary_arithmetic_cpu_kernel.h"
#include <functional>
#include <map>
#include <string>

namespace mindspore {
namespace kernel {
bool BoolBinaryArithmeticCpuKernelMod::Init(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  constexpr size_t kInputNum = 2;
  constexpr size_t kOutputNum = 1;
  CHECK_KERNEL_INPUTS_NUM(inputs.size(), kInputNum, kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(outputs.size(), kOutputNum, kernel_name_);
  return true;
}

int BoolBinaryArithmeticCpuKernelMod::Resize(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  if (auto ret = KernelMod::Resize(inputs, outputs); ret != KRET_OK) {
    return ret;
  }
  return KRET_OK;
}

bool BoolBinaryArithmeticCpuKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &workspace,
                                              const std::vector<KernelTensor *> &outputs) {
  bool *x = GetDeviceAddress<bool>(inputs, kIndex0);
  bool *y = GetDeviceAddress<bool>(inputs, kIndex1);
  bool *out = GetDeviceAddress<bool>(outputs, kIndex0);

  static std::map<std::string, std::function<void(bool *, bool *, bool *)>> bool_impl_map = {
    {"bool_and", [](bool *x, bool *y, bool *out) { *out = static_cast<bool>(*x && *y); }},
    {"bool_or", [](bool *x, bool *y, bool *out) { *out = static_cast<bool>(*x || *y); }},
    {"bool_eq", [](bool *x, bool *y, bool *out) { *out = static_cast<bool>(*x == *y); }},
  };

  if (bool_impl_map.find(kernel_name_) == bool_impl_map.end()) {
    MS_LOG(EXCEPTION) << kernel_name_ << "is not supported on cpu platform.";
  }

  auto func_impl = bool_impl_map.at(kernel_name_);
  func_impl(x, y, out);
  return true;
}

std::vector<KernelAttr> BoolBinaryArithmeticCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {KernelAttr()
                                                   .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                   .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
                                                   .AddOutputAttr(kObjectTypeNumber, kNumberTypeBool)};
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, bool_and, BoolBinaryArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, bool_or, BoolBinaryArithmeticCpuKernelMod);
MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, bool_eq, BoolBinaryArithmeticCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
