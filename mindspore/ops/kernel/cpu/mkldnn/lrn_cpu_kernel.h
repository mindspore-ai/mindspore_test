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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LRN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LRN_CPU_KERNEL_H_

#include <string>
#include <vector>
#include <map>
#include <utility>
#include "kernel/cpu/mkldnn/mkl_cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
class LrnCpuKernelMod : public MKLCpuKernelMod {
 public:
  LrnCpuKernelMod() = default;
  ~LrnCpuKernelMod() override = default;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  bool GetLrnAttr();
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using LrnFunc = std::function<bool(LrnCpuKernelMod *, const std::vector<kernel::KernelTensor *> &,
                                     const std::vector<kernel::KernelTensor *> &)>;

  int64_t depth_radius_{1};
  float bias_{0.0};
  float alpha_{0.0};
  float beta_{0.0};
  LrnFunc kernel_func_;
  dnnl::algorithm dnnl_algorithm_{};
  std::string norm_region_;
  static std::vector<std::pair<KernelAttr, LrnFunc>> func_list_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_LRN_CPU_KERNEL_H_
