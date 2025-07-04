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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_TENSOR_DENSE_MATMUL_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SPARSE_TENSOR_DENSE_MATMUL_CPU_KERNEL_H_

#include <vector>
#include <utility>
#include <map>
#include <string>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace sparse_tensor_dense_matmul_cpu {
class SparseTensorDenseMatmulCpuKernelMod : public NativeCpuKernelMod {
 public:
  SparseTensorDenseMatmulCpuKernelMod() = default;
  ~SparseTensorDenseMatmulCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using SparseTensorDenseMatmulFunc =
    std::function<bool(SparseTensorDenseMatmulCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, SparseTensorDenseMatmulFunc>> func_list_;
  SparseTensorDenseMatmulFunc kernel_func_;
  std::vector<size_t> output_shape_;
  std::vector<size_t> b_shape_;
  size_t output_size_{0};
  size_t values_size_{0};
  bool adj_st_{false};
  bool adj_dt_{false};
  enum input_list_ { INDICES, VALUES, SPARSE_SHAPE, DENSE };
};
}  // namespace sparse_tensor_dense_matmul_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RMSPROP_CPU_KERNEL_H_
