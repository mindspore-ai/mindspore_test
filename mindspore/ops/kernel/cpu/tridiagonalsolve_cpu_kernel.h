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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRIDIAGONAL_SOLVE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRIDIAGONAL_SOLVE_CPU_KERNEL_H_

#include <complex>
#include <memory>
#include <map>
#include <unordered_map>
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace tridiagonalsolve_cpu {
class TridiagonalSolveCPUKernelMod : public NativeCpuKernelMod {
 public:
  TridiagonalSolveCPUKernelMod() = default;
  ~TridiagonalSolveCPUKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool partial_pivoting_{false};
  bool res_{false};
  TypeId diag_dtype_{kTypeUnknown};
  TypeId dtype_{kTypeUnknown};
  int batch_{0};
  int n_{0};
  int32_t diags_size_{0};
  int32_t rhs_size_{0};
  ShapeVector input0_shape;
  ShapeVector input1_shape;

  using TridiagonalSolveFunc =
    std::function<bool(TridiagonalSolveCPUKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, TridiagonalSolveFunc>> func_list_;
  TridiagonalSolveFunc kernel_func_;

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  bool CheckInputValue_(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  bool ChooseDataType_(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                       size_t nth_batch, int i);
  template <typename T>
  bool DoComputeWithPartPivoting_(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs,
                                  size_t nth_batch, int i);
  template <typename T>
  bool DoComputeWithoutPartPivoting_(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs, size_t nth_batch, int i);
};
}  // namespace tridiagonalsolve_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRIDIAGONAL_SOLVE_CPU_KERNEL_H_
