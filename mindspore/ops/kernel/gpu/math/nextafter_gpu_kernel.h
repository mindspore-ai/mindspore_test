/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NEXTAFTER_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NEXTAFTER_GPU_KERNEL_H_
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <algorithm>
#include <functional>
#include <map>
#include "abstract/utils.h"
#include "common/ms_factory.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/gpu_kernel_factory.h"
#include "kernel/gpu/math/nextafter_gpu_kernel.h"
#include "kernel/gpu/cuda_impl/cuda_ops/nextafter_impl.cuh"

namespace mindspore {
namespace kernel {
class NextAfterGpuKernelMod : public NativeGpuKernelMod {
 public:
  NextAfterGpuKernelMod() { ResetResource(); }
  ~NextAfterGpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

  void ResetResource() noexcept {
    unit_size_ = 1;
    input_elements_ = 0;
    is_null_input_ = false;
    output_size_list_.clear();
  }

 protected:
  void InitSizeLists() {
    output_size_list_.clear();
    size_t input_size = input_elements_ * unit_size_;
    output_size_list_.push_back(input_size);
  }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);
  using NextAfterFunc =
    std::function<bool(NextAfterGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

 private:
  size_t unit_size_{1};
  size_t input_elements_;
  NextAfterFunc kernel_func_{};
  std::optional<bool> is_input_dynamic_shape_{};
  bool is_null_input_{false};
  void *cuda_stream_{nullptr};

  static std::vector<std::pair<KernelAttr, NextAfterFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_MULTINOMIAL_GPU_KERNEL_H_
