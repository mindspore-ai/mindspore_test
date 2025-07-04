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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NON_MAX_SUPPRESSION_WITH_OVERLAPS_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NON_MAX_SUPPRESSION_WITH_OVERLAPS_GPU_KERNEL_H_

#include <iostream>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "abstract/utils.h"
#include "kernel/gpu/gpu_kernel.h"
#include "kernel/gpu/kernel_constants.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
class NMSWithOverlapsFwdGpuKernelMod : public NativeGpuKernelMod {
 public:
  NMSWithOverlapsFwdGpuKernelMod() {
    KernelMod::kernel_name_ = "NonMaxSuppressionWithOverlaps";
    num_input_ = 0;
    num_output_ = 0;
    is_null_input_ = false;
    ceil_power_2 = 0;
    data_unit_size_ = 0;
    stream_ptr_ = nullptr;
  }
  ~NMSWithOverlapsFwdGpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs, void *stream_ptr) override {
    return kernel_func_(this, inputs, workspace, outputs, stream_ptr);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 protected:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs, void *stream_ptr);
  using NMSWithOverlapsFunc = std::function<bool(
    NMSWithOverlapsFwdGpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
    const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &, void *)>;
  NMSWithOverlapsFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, NMSWithOverlapsFunc>> func_list_;
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

 private:
  void ResetResource();
  void InitSizeLists();
  void *stream_ptr_;
  int num_input_;
  int num_output_;
  bool is_null_input_;
  // default values
  size_t ceil_power_2;
  size_t data_unit_size_; /* size of T */
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_NON_MAX_SUPPRESSION_WITH_OVERLAPS_GPU_KERNEL_H_
