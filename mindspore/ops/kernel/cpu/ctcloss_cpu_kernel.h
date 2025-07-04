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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTCLOSS_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTCLOSS_CPU_KERNEL_H_

#include <memory>
#include <unordered_map>
#include <vector>
#include <algorithm>
#include <limits>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace ctcloss_cpu {
class CTCLossCpuKernelMod : public NativeCpuKernelMod {
 public:
  CTCLossCpuKernelMod() = default;
  ~CTCLossCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override;

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  void GenLabelWithBlank(const uint32_t *seq_len, const std::vector<std::vector<uint32_t>> &batch_label,
                         std::vector<std::vector<uint32_t>> *label_with_blank) const;

  template <typename T>
  void CalculateFwdVar(const std::vector<uint32_t> &label_with_blank, const std::vector<std::vector<T>> &y,
                       std::vector<std::vector<T>> *log_alpha_b) const;
  template <typename T>
  void CalculateBwdVar(const std::vector<uint32_t> &label_with_blank, const std::vector<std::vector<T>> &y,
                       std::vector<std::vector<T>> *log_beta_b) const;
  template <typename T>
  void CalculateGrad(const std::vector<uint32_t> &label_with_blank, const std::vector<std::vector<T>> &y,
                     const std::vector<std::vector<T>> &log_alpha_b, const std::vector<std::vector<T>> &log_beta_b,
                     const T log_pzx, std::vector<std::vector<T>> *dy) const;

  template <typename T>
  void LaunchKernel(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) const;

  ShapeVector probs_shape_;
  ShapeVector indices_dims_;
  ShapeVector labels_dims_;
  size_t num_class_{0};
  size_t max_time_{0};
  size_t batch_size_{0};
  uint32_t blank_index_{0};
  TypeId dtype_{kTypeUnknown};
  bool preprocess_collapse_repeated_{false};
  bool ctc_merge_repeated_{false};
  bool ignore_longer_outputs_than_inputs_{false};
};
}  // namespace ctcloss_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTCLOSS_CPU_KERNEL_H_
