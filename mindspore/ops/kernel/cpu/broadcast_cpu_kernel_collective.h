/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BROADCAST_CPU_KERNEL_COLLECTIVE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BROADCAST_CPU_KERNEL_COLLECTIVE_H_

#include <vector>

#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace broadcast_cpu {
class BroadcastCPUKernelMod : public NativeCpuKernelMod {
 public:
  BroadcastCPUKernelMod() = default;
  ~BroadcastCPUKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  uint32_t root_rank_ = 0;
  mindspore::TypeId input_dtype_ = kNumberTypeFloat32;
};
}  // namespace broadcast_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_BROADCAST_CPU_KERNEL_COLLECTIVE_H_
