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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEGMENT_MEAN_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEGMENT_MEAN_CPU_KERNEL_H_
#include <vector>
#include <memory>
#include <string>
#include <complex>
#include <utility>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace segment_mean_cpu {
class SegmentMeanCPUKernelMod : public NativeCpuKernelMod {
 public:
  SegmentMeanCPUKernelMod() = default;
  ~SegmentMeanCPUKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  template <typename T1, typename T2>
  bool LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs);

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  std::vector<int64_t> input_x_shape_{};
  std::vector<int64_t> segment_ids_shape_{};
  std::vector<int64_t> output_shape_{};
  size_t input_x_num_;
  size_t segment_ids_num_;
  size_t output_num_;
  TypeId input_x_dtype_{kTypeUnknown};
  TypeId segment_ids_dtype_{kTypeUnknown};
  TypeId output_dtype_{kTypeUnknown};
};
}  // namespace segment_mean_cpu
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEGMENT_MEAN_CPU_KERNEL_H_
