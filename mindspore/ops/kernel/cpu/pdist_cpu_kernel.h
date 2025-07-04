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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PDIST_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PDIST_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include <functional>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"
#include "mindspore/ops/infer/pdist.h"

namespace mindspore {
namespace kernel {
namespace pdist_cpu {
class PdistCpuKernelMod : public NativeCpuKernelMod {
 public:
  PdistCpuKernelMod() = default;
  ~PdistCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename F, typename T>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  template <typename T>
  void Apply_pdist(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

  int64_t h_;
  int64_t w_;
  float p_;
  TypeId dtype_{kTypeUnknown};
};
}  // namespace pdist_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_PDIST_CPU_KERNEL_H_
