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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RGB_TO_HSV_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RGB_TO_HSV_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <string>
#include <utility>
#include <map>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace rgb_to_hsv_cpu {
class RGBToHSVCpuKernelMod : public NativeCpuKernelMod {
 public:
  RGBToHSVCpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  TypeId input_dtype{kTypeUnknown};
  size_t input0_elements_nums_;
  bool res_;

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  template <typename T>
  bool ComputeFloat(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  bool ComputeHalf(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  using RGBToHSVFunc = std::function<bool(RGBToHSVCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                          const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, RGBToHSVFunc>> func_list_;
  RGBToHSVFunc kernel_func_;
};
}  // namespace rgb_to_hsv_cpu
}  // namespace kernel
}  // namespace mindspore
#endif
