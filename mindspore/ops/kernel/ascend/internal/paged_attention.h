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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PAGED_ATTENTION_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PAGED_ATTENTION_H_

#include <string>
#include <vector>
#include <utility>

#include "kernel/ascend/internal/internal_kernel_mod.h"
#include "include/internal.h"

namespace mindspore {
namespace kernel {
class InternalPagedAttention : public InternalKernelMod {
 public:
  InternalPagedAttention() : InternalKernelMod() {}
  ~InternalPagedAttention() = default;

 protected:
  bool Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs,
                                       const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) override;
  bool UpdateParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) override;
  uint64_t GenerateTilingKey(const std::vector<KernelTensor *> &inputs) override;

 private:
  inline void CheckMask() {
    param_.mask_type = internal::PagedAttentionParam::MaskType::kMaskTypeNone;
    auto enable_lookahead =
      std::any_of(param_.q_seq_len.begin(), param_.q_seq_len.end(), [](int32_t seq_len) { return seq_len > 1; });
    if (enable_lookahead) {
      if (has_attn_mask_) {
        param_.mask_type = internal::PagedAttentionParam::MaskType::kMaskTypeLookAhead;
      }
    } else {
      param_.q_seq_len.clear();
    }

    if (has_alibi_mask_) {
      if (param_.mask_type == internal::PagedAttentionParam::MaskType::kMaskTypeLookAhead) {
        MS_LOG(EXCEPTION) << "For op " << kernel_name_ << ", lookahead cannot be enabled when alibi_mask exists.";
      } else {
        param_.mask_type = internal::PagedAttentionParam::MaskType::kMaskTypeAlibi;
      }
    }
  }

  internal::PagedAttentionParam param_;
  bool created_flag_{false};
  bool has_attn_mask_{false};
  bool has_alibi_mask_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PAGED_ATTENTION_H_
