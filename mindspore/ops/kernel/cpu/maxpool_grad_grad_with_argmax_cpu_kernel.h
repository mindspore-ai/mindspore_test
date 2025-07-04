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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAX_POOL_GRAD_GRAD_WITH_ARGMAX_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAX_POOL_GRAD_GRAD_WITH_ARGMAX_CPU_KERNEL_H_

#include <string>
#include <vector>
#include <memory>
#include <map>
#include <utility>
#include <unordered_map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace maxpool_grad_grad_with_argmax_cpu {
class MaxPoolGradGradWithArgmaxCpuKernelMod : public NativeCpuKernelMod,
                                              public MatchKernelHelper<MaxPoolGradGradWithArgmaxCpuKernelMod> {
 public:
  MaxPoolGradGradWithArgmaxCpuKernelMod() = default;
  ~MaxPoolGradGradWithArgmaxCpuKernelMod() = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T, typename I>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                    const std::vector<kernel::KernelTensor *> &outputs);

  size_t output_elements_ = 0;
  size_t input_batch_stride_ = 0;
  size_t output_batch_stride_ = 0;
};
}  // namespace maxpool_grad_grad_with_argmax_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_MAX_POOL_GRAD_GRAD_WITH_ARGMAX_CPU_KERNEL_H_
