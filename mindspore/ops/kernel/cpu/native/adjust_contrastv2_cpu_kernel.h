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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ADJUST_CONTRASTV2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ADJUST_CONTRASTV2_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "kernel/cpu/cpu_kernel.h"
#include "include/runtime/hardware_abstract/kernel_base/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace adjust_contrastv2_cpu {
constexpr size_t MIN_DIM = 3;

class AdjustContrastv2CpuKernelMod : public NativeCpuKernelMod {
 public:
  AdjustContrastv2CpuKernelMod() = default;
  ~AdjustContrastv2CpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs);

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchAdjustContrastv2Kernel(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs);

  std::vector<int64_t> images_shape_;
  TypeId input_type_{kTypeUnknown};
};
}  // namespace adjust_contrastv2_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_ADJUST_CONTRASTV2_CPU_KERNEL_H_
