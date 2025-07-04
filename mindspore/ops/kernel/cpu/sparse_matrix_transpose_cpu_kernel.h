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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_MATRIX_TRANSPOSE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_MATRIX_TRANSPOSE_CPU_KERNEL_H_

#include <algorithm>
#include <complex>
#include <iostream>
#include <map>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace sparse_matrix_transpose_cpu {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class SparseMatrixTransposeCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseMatrixTransposeCpuKernelMod() = default;
  ~SparseMatrixTransposeCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename indiceT, typename valueT>
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) const;

  template <typename indiceT, typename valueT>
  void LaunchcomplexKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) const;

  size_t x_batch_pointers_size_;
  size_t x_value_size_;
  size_t x_col_indice_size_;
  size_t x_row_pointer_size_;
  size_t rank_x_;
  bool conjugate;
  TypeId indiceT_{kTypeUnknown};
  TypeId valueT_{kTypeUnknown};
};
}  // namespace sparse_matrix_transpose_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_MATRIX_TRANSPOSE_CPU_KERNEL_H_
