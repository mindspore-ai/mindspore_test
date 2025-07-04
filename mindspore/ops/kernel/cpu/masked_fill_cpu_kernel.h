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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MASKED_FILL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MASKED_FILL_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace masked_fill_cpu {
class MaskedFillCpuKernelMod : public NativeCpuKernelMod {
 public:
  MaskedFillCpuKernelMod() = default;
  ~MaskedFillCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using MaskedFillFunc = std::function<bool(MaskedFillCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                            const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, MaskedFillFunc>> func_list_;
  MaskedFillFunc kernel_func_;
  size_t output_size_{1};
  size_t inner_size_{1};
  size_t value_size_{1};
  int64_t batch_rank_{0};
  std::vector<size_t> mask_index_;
  std::vector<size_t> input_index_;
  bool need_broadcast_{false};
};
}  // namespace masked_fill_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MASKED_FILL_CPU_KERNEL_H_
