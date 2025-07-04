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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NLLLOSS_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NLLLOSS_GRAD_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"
#include "nnacl/kernel/nllloss.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {
namespace nllloss_grad_cpu {
class NLLLossGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  NLLLossGradCpuKernelMod() = default;
  ~NLLLossGradCpuKernelMod() override = default;

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
  using NLLLossGradFunc =
    std::function<bool(NLLLossGradCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;
  NLLLossGradFunc kernel_func_;
  static std::vector<std::pair<KernelAttr, NLLLossGradFunc>> func_list_;

 private:
  NLLLossStruct nllloss_param_{};
  Reduction reduction_type_;
  int64_t ignore_index_;
};
}  // namespace nllloss_grad_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_NLLLOSS_GRAD_CPU_KERNEL_H_
