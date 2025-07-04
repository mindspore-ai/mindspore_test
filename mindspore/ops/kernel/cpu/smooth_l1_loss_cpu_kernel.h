/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICES_CPU_KERNEL_SMOOTH_L1_LOSS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICES_CPU_KERNEL_SMOOTH_L1_LOSS_CPU_KERNEL_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <utility>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "mindapi/base/types.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace smooth_l1_loss_cpu {
class SmoothL1LossCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<SmoothL1LossCpuKernelMod> {
 public:
  SmoothL1LossCpuKernelMod() = default;
  ~SmoothL1LossCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

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

  template <typename T>
  void CalElements(const T *predict_addr, const T *target_addr, T *result_addr);

  float beta_{1.0};
  TypeId dtype_{kTypeUnknown};
  int64_t tensor_size_{1};
  Reduction reduction_{Reduction::MEAN};
};
}  // namespace smooth_l1_loss_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICES_CPU_KERNEL_SMOOTH_L1_LOSS_CPU_KERNEL_H_
