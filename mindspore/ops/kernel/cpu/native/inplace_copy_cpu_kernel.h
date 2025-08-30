/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_INPLACE_COPY_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_INPLACE_COPY_CPU_KERNEL_H_

#include <map>
#include <vector>
#include <utility>
#include "kernel/cpu/cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace inplace_copy_cpu {
class InplaceCopyCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<InplaceCopyCpuKernelMod> {
 public:
  InplaceCopyCpuKernelMod() = default;
  ~InplaceCopyCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename S, typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);
  template <typename T>
  void InplaceCopySameDtypeSameShape(T *input, T *output, size_t input_size, size_t output_size);
  template <typename T>
  void InplaceCopyBroadcastTo(T *input, T *output, const std::vector<int64_t> &input_shape,
                              const std::vector<int64_t> &output_shape);
  std::vector<int64_t> self_shape_ = {};
  std::vector<int64_t> value_shape_ = {};
  TypeId self_dtype_;
  TypeId value_dtype_;
  int mode_;
  bool is_empty_;
};
}  // namespace inplace_copy_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_INPLACE_COPY_CPU_KERNEL_H_
