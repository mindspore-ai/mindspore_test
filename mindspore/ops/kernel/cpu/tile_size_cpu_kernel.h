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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TILE_SIZE_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TILE_SIZE_CPU_KERNEL_H_
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "mindspore/ops/infer/tile_size.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace tile_size_cpu {
class TileSizeCpuKernelMod : public NativeCpuKernelMod {
 public:
  TileSizeCpuKernelMod() = default;
  ~TileSizeCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs,
                    const std::vector<kernel::KernelTensor *> &workspace) const;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

  using TileSizeFunc =
    std::function<bool(TileSizeCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                       const std::vector<kernel::KernelTensor *> &, const std::vector<kernel::KernelTensor *> &)>;

  static std::vector<std::pair<KernelAttr, TileSizeFunc>> func_list_;
  TileSizeFunc kernel_func_;
};
}  // namespace tile_size_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_TILE_SIZE_CPU_KERNEL_H_
