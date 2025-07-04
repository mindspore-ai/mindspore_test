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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_SPARSE_MATRIX_ORDERING_AMD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_SPARSE_MATRIX_ORDERING_AMD_CPU_KERNEL_H_
#define EIGEN_USE_THREADS
#define EIGEN_USE_SIMPLE_THREAD_POOL

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
class SparseMatrixOrderingAMDCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseMatrixOrderingAMDCpuKernelMod() = default;
  ~SparseMatrixOrderingAMDCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);

  using SparseMatrixOrderingAMDLaunchFunc =
    std::function<bool(SparseMatrixOrderingAMDCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, SparseMatrixOrderingAMDLaunchFunc>> func_list_;
  SparseMatrixOrderingAMDLaunchFunc kernel_func_;

  std::vector<int64_t> x_dense_shape_shape_{0};
  std::vector<int64_t> x_batch_pointers_shape_{0};
  std::vector<int64_t> x_row_pointers_shape_{0};
  std::vector<int64_t> x_col_indices_shape_{0};
  std::vector<int64_t> x_values_shape_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_EIGEN_SPARSE_MATRIX_ORDERING_AMD_CPU_KERNEL_H_
