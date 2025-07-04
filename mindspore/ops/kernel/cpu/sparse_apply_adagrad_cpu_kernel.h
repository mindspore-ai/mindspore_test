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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_ADAGRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_ADAGRAD_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include <utility>
#include "kernel/cpu/sparse_optimizer_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace sparse_apply_adagrad_cpu {
class SparseApplyAdagradCpuKernelMod : public sparse_optimizer_cpu::SparseOptimizerCpuKernelMod,
                                       public MatchKernelHelper<SparseApplyAdagradCpuKernelMod> {
 public:
  SparseApplyAdagradCpuKernelMod() { ResetResource(); }
  ~SparseApplyAdagradCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 protected:
  void ResetResource() noexcept;

 protected:
  float lr_;
  bool update_slots_{true};

 private:
  int64_t batch_rank_{0};
  int64_t batch_size_{1};
  size_t indices_inner_size_{0};
  size_t grad_inner_size_{0};
  size_t var_inner_size_{0};
  template <typename T>
  void InitWorkspaceSize();

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &) const;
};
}  // namespace sparse_apply_adagrad_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_APPLY_ADAGRAD_CPU_KERNEL_H_
