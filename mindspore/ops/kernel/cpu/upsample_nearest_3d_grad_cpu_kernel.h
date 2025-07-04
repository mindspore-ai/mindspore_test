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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_NEAREST_3D_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_NEAREST_3D_GRAD_CPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "common/common_utils.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace upsample_nearest_3d_grad_cpu {
class UpsampleNearest3DGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  UpsampleNearest3DGradCpuKernelMod() = default;
  ~UpsampleNearest3DGradCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

  std::vector<size_t> GetLaunchIgnoredInputAddressIdx() const override { return {kIndex2}; }

 private:
  void ComputeNearestIndex(int64_t *const indices, const int64_t stride, const int64_t input_szie,
                           const int64_t output_size, const double scale) const;

  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using KernelRunFunc = std::function<bool(UpsampleNearest3DGradCpuKernelMod *, const std::vector<KernelTensor *> &,
                                           const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;
  KernelRunFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, KernelRunFunc>> func_list_;
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<int64_t> none_list_;
  std::vector<float> scales_;
};
}  // namespace upsample_nearest_3d_grad_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_NEAREST_3D_GRAD_CPU_KERNEL_H_
