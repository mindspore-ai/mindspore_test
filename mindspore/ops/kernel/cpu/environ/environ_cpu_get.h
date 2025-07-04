/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ENVIRON_ENVIRON_CPU_GET_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ENVIRON_ENVIRON_CPU_GET_H_

#include <vector>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
class EnvironGetCpuKernelMod : public NativeCpuKernelMod {
 public:
  EnvironGetCpuKernelMod() : value_type_attr_(kObjectTypeTensorType), handle_size_(0), key_size_(0), value_size_(0) {}
  ~EnvironGetCpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  }

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddOutputAttr(kNumberTypeInt64),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddOutputAttr(kNumberTypeInt32),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt16)
                                                           .AddOutputAttr(kNumberTypeInt16),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeUInt32)
                                                           .AddOutputAttr(kNumberTypeUInt32),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeUInt16)
                                                           .AddOutputAttr(kNumberTypeUInt16),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeUInt8)
                                                           .AddOutputAttr(kNumberTypeUInt8),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeUInt64)
                                                           .AddOutputAttr(kNumberTypeUInt64),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeFloat32)
                                                           .AddOutputAttr(kNumberTypeFloat32),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeFloat16)
                                                           .AddOutputAttr(kNumberTypeFloat16),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeBool)
                                                           .AddOutputAttr(kNumberTypeBool)};
    return support_list;
  }

 private:
  // The type of env tensor get.
  TypeId value_type_attr_;

  size_t handle_size_;
  size_t key_size_;
  size_t value_size_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_ENVIRON_ENVIRON_CPU_GET_H_
