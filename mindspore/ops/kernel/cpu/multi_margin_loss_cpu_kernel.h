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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTI_MARGIN_LOSS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTI_MARGIN_LOSS_CPU_KERNEL_H_
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <map>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace multi_margin_loss_cpu {
class MultiMarginLossCPUKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<MultiMarginLossCPUKernelMod> {
 public:
  MultiMarginLossCPUKernelMod() = default;

  ~MultiMarginLossCPUKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    MS_EXCEPTION_IF_NULL(kernel_func_);
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                    const std::vector<KernelTensor *> &outputs);

 private:
  template <typename T>
  void LaunchKernelFP32AndFP64(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);
  template <typename T>
  void LaunchKernelFP16(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  void CheckParam(const CNodePtr &kernel_node);
  bool weight_defined_{false};
  size_t batch_size_ = 2;
  size_t dims_ = 1;
  std::string reduction_ = MEAN;
  float margin_ = 1.0;
  int64_t p_ = 1;
  size_t input_num_ = 1;
  TypeId dtype_{kTypeUnknown};
};
}  // namespace multi_margin_loss_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_MULTI_MARGIN_LOSS_CPU_KERNEL_H_
