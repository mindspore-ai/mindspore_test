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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PAD_AND_SHIFT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PAD_AND_SHIFT_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <unordered_map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace pad_and_shift_cpu {
class PadAndShiftCpuKernelMod : public NativeCpuKernelMod {
 public:
  PadAndShiftCpuKernelMod() = default;
  ~PadAndShiftCpuKernelMod() override = default;
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override {
    return true;
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  template <typename T>
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<kernel::KernelTensor *> &outputs);

  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }

  std::vector<KernelAttr> GetOpSupport() override {
    static const std::vector<KernelAttr> support_list = {KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddInputAttr(kNumberTypeInt32)
                                                           .AddOutputAttr(kNumberTypeInt32),
                                                         KernelAttr()
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddInputAttr(kNumberTypeInt64)
                                                           .AddOutputAttr(kNumberTypeInt64)};
    return support_list;
  }

 private:
  size_t batch_size_{1};
  size_t cum_sum_size_{1};
  size_t type_size_{4};
  size_t output_size_{1};
  ShapeVector input_shape_;
  TypeId input_x_dtype_{kTypeUnknown};
};
}  // namespace pad_and_shift_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PAD_AND_SHIFT_CPU_KERNEL_H_
