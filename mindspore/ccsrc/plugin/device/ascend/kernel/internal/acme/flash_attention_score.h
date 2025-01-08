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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_ACME_FLASH_ATTENTION_SCORE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_ACME_FLASH_ATTENTION_SCORE_H_

#include <string>
#include <vector>
#include <utility>

#include "plugin/device/ascend/kernel/internal/acme_kernel_mod.h"
#include "acme/include/acme.h"

namespace mindspore {
namespace kernel {
class AcmeFlashAttentionScore : public AcmeKernelMod {
 public:
  AcmeFlashAttentionScore() : AcmeKernelMod() {}
  ~AcmeFlashAttentionScore() = default;

 protected:
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  acme::AcmeOpPtr CreateKernel(const acme::InputsImmutableInfoList &inputs,
                               const acme::OutputsImmutableInfoList &outputs,
                               const std::vector<KernelTensor *> &ms_inputs,
                               const std::vector<KernelTensor *> &ms_outputs) override;
  void *GetParam() override { return &param_; }
  uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs) override;
  bool IsNeedRecreate(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;

 private:
  acme::FlashAttentionScoreParam param_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_ACME_FLASH_ATTENTION_SCORE_H_
