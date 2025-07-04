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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_POLAR_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_POLAR_GPU_KERNEL_H_
#include <cuda_runtime_api.h>
#include <cuda_runtime.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include <algorithm>
#include <functional>
#include "mindspore/ops/infer/ops_func_impl/polar.h"
#include "abstract/utils.h"
#include "common/ms_factory.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/cuda_impl/cuda_ops/complex.h"
#include "kernel/gpu/cuda_impl/cuda_ops/polar_impl.cuh"

namespace mindspore {
namespace kernel {
class PolarGpuKernelMod : public NativeGpuKernelMod {
 public:
  PolarGpuKernelMod() { ResetResource(); }
  ~PolarGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  void ResetResource() noexcept {
    output_elements_ = 0;
    is_null_input_ = false;
    output_size_list_.clear();
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  using Polarfunc =
    std::function<bool(PolarGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

 private:
  size_t output_elements_;
  size_t unit_input_size_{1};
  size_t unit_output_size_{1};
  Polarfunc kernel_func_{};
  bool is_null_input_{false};
  void *cuda_stream_{nullptr};
  static std::vector<std::pair<KernelAttr, Polarfunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_MATH_POLAR_GPU_KERNEL_H_
