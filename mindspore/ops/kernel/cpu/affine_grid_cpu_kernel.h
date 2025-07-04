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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_AFFINEGRID_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_AFFINEGRID_CPU_KERNEL_H_

#include <Eigen/Dense>
#include <vector>
#include <unordered_map>
#include <utility>
#include <map>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace affine_grid_cpu {
using complex64 = std::complex<float>;
using complex128 = std::complex<double>;

class AffineGridCpuKernelMod : public NativeCpuKernelMod {
 public:
  AffineGridCpuKernelMod() = default;
  ~AffineGridCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);
  template <typename T>
  void LaunchKernel_3D(const std::vector<kernel::KernelTensor *> &inputs,
                       const std::vector<kernel::KernelTensor *> &outputs);
  template <typename T>
  void LaunchKernel_4D(const std::vector<kernel::KernelTensor *> &inputs,
                       const std::vector<kernel::KernelTensor *> &outputs);
  using AffineGridFunc =
    std::function<bool(AffineGridCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, AffineGridFunc>> func_list_;
  AffineGridFunc kernel_func_;

  TypeId output_type_;
  std::vector<int64_t> output_size_dims_;
  bool align_corners_{false};
  std::vector<KernelTensor *> outputs_;
  ShapeVector output_shape_;
};
}  // namespace affine_grid_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_AFFINEGRID_CPU_KERNEL_H_
