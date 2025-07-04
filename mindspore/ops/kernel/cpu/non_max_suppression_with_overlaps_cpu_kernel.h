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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NON_MAX_SUPPRESSION_WITH_OVERLAPS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NON_MAX_SUPPRESSION_WITH_OVERLAPS_CPU_KERNEL_H_

#include <vector>
#include <memory>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace non_max_suppression_with_overlaps_cpu {
class NonMaxSuppressionWithOverlapsCpuKernelMod : public NativeCpuKernelMod {
 public:
  NonMaxSuppressionWithOverlapsCpuKernelMod() = default;
  ~NonMaxSuppressionWithOverlapsCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

 protected:
  bool IsNeedUpdateOutputShapeAndSize() override { return true; }
  void UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  int num_boxes_{0};
  int64_t real_output_size_{0};  // Dynamic shape related.
};
}  // namespace non_max_suppression_with_overlaps_cpu
};  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_CPU_KERNEL_NON_MAX_SUPPRESSION_WITH_OVERLAPS_CPU_KERNEL_H_
