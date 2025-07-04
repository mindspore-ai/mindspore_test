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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTC_GREEDY_DECODER_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTC_GREEDY_DECODER_CPU_KERNEL_H_

#include <memory>
#include <vector>
#include <utility>
#include <map>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "common/ms_factory.h"

namespace mindspore {
namespace kernel {
namespace ctc_greedy_decoder_cpu {
class CTCGreedyDecoderCpuKernelMod : public NativeCpuKernelMod, public MatchKernelHelper<CTCGreedyDecoderCpuKernelMod> {
 public:
  CTCGreedyDecoderCpuKernelMod() = default;
  ~CTCGreedyDecoderCpuKernelMod() override = default;

  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  bool Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
              const std::vector<KernelTensor *> &outputs) override {
    return kernel_func_(this, inputs, workspace, outputs);
  }

  int Resize(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  bool IsNeedUpdateOutputShapeAndSize() override { return true; }

 private:
  bool merge_repeated_{true};
  std::vector<TypeId> types_;
  int64_t max_time_;
  int64_t batch_size_;
  int64_t num_classes_raw_;

  template <typename T>
  bool LaunchKernel(const std::vector<kernel::KernelTensor *> &inputs, const std::vector<KernelTensor *> &,
                    const std::vector<kernel::KernelTensor *> &outputs);
};
}  // namespace ctc_greedy_decoder_cpu
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_CTC_GREEDY_DECODER_CPU_KERNEL_H_
