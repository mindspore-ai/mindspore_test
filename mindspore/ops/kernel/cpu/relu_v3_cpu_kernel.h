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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RELU_V3_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RELU_V3_CPU_KERNEL_H_

#include <memory>
#include <string>
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore::kernel {
namespace relu_v3_cpu {
constexpr auto kUnknown = "Unknown";

class ReLUV3CpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<ReLUV3CpuKernelMod> {
 public:
  ReLUV3CpuKernelMod() = default;
  explicit ReLUV3CpuKernelMod(const std::string &kernel_type) : kernel_type_(kernel_type) {}
  ~ReLUV3CpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                    const std::vector<kernel::KernelTensor *> &outputs);

  std::string kernel_type_{kUnknown};
};
}  // namespace relu_v3_cpu
}  // namespace mindspore::kernel

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_RELU_V3_CPU_KERNEL_H_
