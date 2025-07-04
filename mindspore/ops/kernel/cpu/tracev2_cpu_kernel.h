/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRACEV2_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRACEV2_CPU_KERNEL_H_

#include <vector>
#include <complex>
#include <utility>
#include <string>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace tracev2_cpu {
class TraceV2CpuKernelMod : public NativeCpuKernelMod {
 public:
  TraceV2CpuKernelMod() = default;
  ~TraceV2CpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T_in, typename T_out>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                    const std::vector<kernel::KernelTensor *> &outputs);
  using TraceV2Func = std::function<bool(TraceV2CpuKernelMod *, const std::vector<KernelTensor *> &,
                                         const std::vector<KernelTensor *> &, const std::vector<KernelTensor *> &)>;
  static std::vector<std::pair<KernelAttr, TraceV2Func>> func_list_;
  TraceV2Func kernel_func_;

  int64_t offset_{0};
  size_t x_size_{0};
  int64_t mat_row_size_{0};
  int64_t mat_col_size_{0};
  int64_t mat_size_{0};
  int64_t batch_size_{1};
  size_t data_unit_size_{0};
  std::vector<size_t> tanspose_index_;
};
}  // namespace tracev2_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_TRACEV2_CPU_KERNEL_H_
