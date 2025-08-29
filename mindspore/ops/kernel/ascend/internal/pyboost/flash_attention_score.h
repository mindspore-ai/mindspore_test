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
#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_FLASH_ATTENTION_SCORE_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_FLASH_ATTENTION_SCORE_H_

#include <memory>
#include <string>
#include <vector>
#include <utility>

#include "kernel/ascend/internal/pyboost/internal_kernel_info.h"

namespace mindspore {
namespace kernel {
class FlashAttentionScore : public InternalKernelInfo {
 public:
  explicit FlashAttentionScore(std::string &&kernel_name) : InternalKernelInfo(std::move(kernel_name)) {}
  ~FlashAttentionScore() = default;

  void Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key, const uint64_t &tiling_key,
            const TensorPtr &query, const TensorPtr &key, const TensorPtr &value,
            const std::optional<TensorPtr> &real_shift, const std::optional<TensorPtr> &drop_mask,
            const std::optional<TensorPtr> &padding_mask, const std::optional<TensorPtr> &attn_mask,
            const std::vector<int64_t> &prefix, const std::vector<int64_t> &actual_seq_len,
            const std::vector<int64_t> &actual_seq_kvlen, const int64_t &head_num, const float &keep_prob,
            const float &scale_value, const int64_t &pre_tokens, const int64_t &next_tokens,
            const int64_t &inner_precise, const int64_t &input_layout, const int64_t &sparse_mode);

 protected:
  uint64_t GetOrGenerateOpTilingKey(const uint64_t &tiling_key) const override;
  bool UpdateParam() override;
  internal::InternalOpPtr CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                       const internal::OutputsImmutableInfoList &outputs) override;

 private:
  internal::FlashAttentionScoreParam param_;
  bool created_flag_{false};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_INTERNAL_INTERNAL_PYBOOST_FLASH_ATTENTION_SCORE_H_
