/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_PAGED_ATTENTION_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_PAGED_ATTENTION_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "kernel/ascend/internal/pyboost/internal_kernel_info.h"

namespace mindspore {
namespace kernel {
class PagedAttention : public InternalKernelInfo {
 public:
  explicit PagedAttention(std::string &&kernel_name) : InternalKernelInfo(std::move(kernel_name)) {}
  ~PagedAttention() = default;

  void Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key, const uint64_t &tiling_key,
            const TensorPtr &query, const TensorPtr &key_cache, const std::optional<TensorPtr> &value_cache,
            const std::optional<TensorPtr> &block_tabels, const std::optional<TensorPtr> &context_lens,
            const std::optional<TensorPtr> &antiquant_scale, const std::optional<TensorPtr> &antiquant_offset,
            const std::optional<TensorPtr> &attn_mask, const std::optional<TensorPtr> &q_seq_lens,
            const std::optional<TensorPtr> &alibi_mask, const int64_t &head_num, const float &scale_value,
            const int64_t &kv_head_num, const int64_t &kv_cache_quant_mode, const int64_t &mask_mode,
            const int64_t &mla_v_dim);

 protected:
  uint64_t GetOrGenerateOpTilingKey(const uint64_t &tiling_key) const override;
  bool UpdateParam() override;
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;

 private:
  inline bool GetSeqLenFromInputTensor(const TensorPtr &input_tensor, std::vector<int32_t> *seq_len) {
    if (input_tensor == nullptr) {
      return false;
    }

    auto tensor_on_cpu = input_tensor->cpu();
    auto input_tensor_value = static_cast<int32_t *>(tensor_on_cpu->data_c());
    auto input_tensor_value_num = tensor_on_cpu->Size() / sizeof(int32_t);
    seq_len->clear();
    for (size_t i = 0; i < input_tensor_value_num; i++) {
      (*seq_len).emplace_back(input_tensor_value[i]);
    }
    return true;
  }

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
  bool has_attn_mask_{false};
  bool has_alibi_mask_{false};
  bool created_flag_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_PAGED_ATTENTION_H_
