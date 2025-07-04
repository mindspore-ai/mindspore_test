/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_L2_NORMALIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_L2_NORMALIZE_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace l2_normalize_cpu {
class L2NormalizeCpuKernelMod : public NativeCpuKernelMod {
 public:
  L2NormalizeCpuKernelMod() = default;
  ~L2NormalizeCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return func_obj_->RunFunc(inputs, workspace, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::shared_ptr<CpuKernelFunc> func_obj_;
};
}  // namespace l2_normalize_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_L2_NORMALIZE_CPU_KERNEL_H_
