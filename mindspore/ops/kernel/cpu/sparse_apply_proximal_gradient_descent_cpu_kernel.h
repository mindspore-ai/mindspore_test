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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_PROXIMAL_GRADIENT_DESCENT_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_PROXIMAL_GRADIENT_DESCENT_CPU_KERNEL_H_

#include <utility>
#include <vector>
#include <map>

#include "mindspore/ops/infer/sparse_apply_proximal_gradient_descent.h"
#include "kernel/cpu/sparse_optimizer_cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace sparse_apply_proximal_gradient_descent_cpu {
class SparseApplyProximalGradientDescentCpuKernelMod
    : public sparse_optimizer_cpu::SparseOptimizerCpuKernelMod,
      public MatchKernelHelper<SparseApplyProximalGradientDescentCpuKernelMod> {
 public:
  SparseApplyProximalGradientDescentCpuKernelMod() = default;
  ~SparseApplyProximalGradientDescentCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  template <typename I, typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs) const;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }
  void ResetResouce() noexcept;
};
}  // namespace sparse_apply_proximal_gradient_descent_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_PROXIMAL_GRADIENT_DESCENT_CPU_KERNEL_H_
