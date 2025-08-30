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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MIN_DIM_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MIN_DIM_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <memory>
#include <algorithm>

#include "kernel/cpu/native/argmin_with_value_cpu_kernel.h"
#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace min_dim_cpu {
class MinDimCpuKernelMod : public argmin_with_value_cpu::ArgMinWithValueCpuKernelMod {
 public:
  MinDimCpuKernelMod() : argmin_with_value_cpu::ArgMinWithValueCpuKernelMod(1, 0) {}
  ~MinDimCpuKernelMod() override = default;

 protected:
  std::vector<KernelAttr> GetOpSupport() override {
    static std::vector<KernelAttr> kernel_attr_list = {
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
        .AddOutputAttr(kNumberTypeFloat64)
        .AddOutputAttr(kNumberTypeInt64),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat32)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
        .AddOutputAttr(kNumberTypeFloat32)
        .AddOutputAttr(kNumberTypeInt64),
      KernelAttr()
        .AddInputAttr(kNumberTypeFloat16)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
        .AddOutputAttr(kNumberTypeFloat16)
        .AddOutputAttr(kNumberTypeInt64),
      KernelAttr()
        .AddInputAttr(kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
        .AddOutputAttr(kNumberTypeInt64)
        .AddOutputAttr(kNumberTypeInt64),
      KernelAttr()
        .AddInputAttr(kNumberTypeInt32)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
        .AddOutputAttr(kNumberTypeInt32)
        .AddOutputAttr(kNumberTypeInt64),
      KernelAttr()
        .AddInputAttr(kNumberTypeInt16)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
        .AddOutputAttr(kNumberTypeInt16)
        .AddOutputAttr(kNumberTypeInt64),
      KernelAttr()
        .AddInputAttr(kNumberTypeInt8)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
        .AddOutputAttr(kNumberTypeInt8)
        .AddOutputAttr(kNumberTypeInt64),
      KernelAttr()
        .AddInputAttr(kNumberTypeUInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
        .AddOutputAttr(kNumberTypeUInt64)
        .AddOutputAttr(kNumberTypeInt64),
      KernelAttr()
        .AddInputAttr(kNumberTypeUInt32)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
        .AddOutputAttr(kNumberTypeUInt32)
        .AddOutputAttr(kNumberTypeInt64),
      KernelAttr()
        .AddInputAttr(kNumberTypeUInt16)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
        .AddOutputAttr(kNumberTypeUInt16)
        .AddOutputAttr(kNumberTypeInt64),
      KernelAttr()
        .AddInputAttr(kNumberTypeUInt8)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeInt64)
        .AddInputAttr(kObjectTypeNumber, kNumberTypeBool)
        .AddOutputAttr(kNumberTypeUInt8)
        .AddOutputAttr(kNumberTypeInt64),
    };
    return kernel_attr_list;
  }
};
}  // namespace min_dim_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MIN_DIM_CPU_KERNEL_H_
