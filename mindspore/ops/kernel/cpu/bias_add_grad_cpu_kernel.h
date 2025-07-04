/**
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_BIAS_ADD_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_BIAS_ADD_GRAD_CPU_KERNEL_H_

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"
#include "mindspore/ops/infer/ops_func_impl/bias_add_grad.h"

namespace mindspore {
namespace kernel {
namespace bias_add_grad_cpu {
class BiasAddGradCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<BiasAddGradCpuKernelMod> {
 public:
  BiasAddGradCpuKernelMod() = default;
  ~BiasAddGradCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<KernelTensor *> &outputs);

  std::vector<size_t> input_shape_;
  int64_t data_format_ = Format::NCHW;
};
}  // namespace bias_add_grad_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_BIAS_ADD_GRAD_CPU_KERNEL_H_
