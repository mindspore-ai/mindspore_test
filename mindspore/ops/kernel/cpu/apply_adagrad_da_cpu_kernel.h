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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_APPLY_ADAGRAD_DA_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_APPLY_ADAGRAD_DA_CPU_KERNEL_H_

#include <thread>
#include <vector>
#include <map>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace apply_adagrad_da_cpu {
class ApplyAdagradDACpuKernelMod : public NativeCpuKernelMod {
 public:
  ApplyAdagradDACpuKernelMod() = default;
  ~ApplyAdagradDACpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  template <typename T>
  void LaunchApplyAdagradDA(T *var, T *gradient_accumulator, T *gradient_squared_accumulator, const T *grad,
                            const T *lr, const T *l1, const T *l2, const int *global_step, size_t start,
                            size_t end) const;
  int64_t batch_size_{1};
  int64_t batch_rank_{0};
  int64_t input_elements_;
  TypeId dtype_{kTypeUnknown};
};
}  // namespace apply_adagrad_da_cpu
}  // namespace kernel
}  // namespace mindspore
#endif
