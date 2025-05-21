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

#include "plugin/device/ascend/kernel/internal/pyboost/paged_attention.h"

#include <memory>
#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalKernelInfoPagedAttention::CreateKernel(
  const internal::InputsImmutableInfoList &inputs, const internal::OutputsImmutableInfoList &outputs) {
  internal::PagedAttentionParam param;
  param.head_num = head_num_;
  param.kv_head_num = kv_head_num_;
  param.tor = tor_;
  param.kv_cache_quant_mode = kv_cache_quant_mode_;
  param.mask_mode = mask_mode_;
  param.mla_v_dim = mla_v_dim_;

  CheckMask();
  param.mask_type = mask_type_;
  param.q_seq_len = q_seq_len_;
  param.kv_seq_len = kv_seq_len_;

  return internal::CreatePagedAttentionOp(inputs, outputs, param, internal::kInternalPagedAttentionOpName);
}

uint64_t InternalKernelInfoPagedAttention::GenerateTilingKey(const std::string &kernel_name,
                                                             const std::vector<BaseTensorPtr> &inputs) {
  return CalcInternalOpTilingHash(kernel_name, inputs, q_seq_len_, kv_seq_len_);
}

void InternalKernelInfoPagedAttention::Call(
  const std::shared_ptr<pyboost::OpRunner> &op, const BaseTensorPtr &query, const BaseTensorPtr &key_cache,
  const std::optional<BaseTensorPtr> &value_cache, const std::optional<BaseTensorPtr> &block_tabels,
  const std::optional<BaseTensorPtr> &context_lens, const std::optional<BaseTensorPtr> &antiquant_scale,
  const std::optional<BaseTensorPtr> &antiquant_offset, const std::optional<BaseTensorPtr> &attn_mask,
  const std::optional<BaseTensorPtr> &q_seq_lens, const std::optional<BaseTensorPtr> &alibi_mask,
  const int64_t &head_num, const float &scale_value, const int64_t &kv_head_num, const int64_t &kv_cache_quant_mode,
  const int64_t &mask_mode, const int64_t &mla_v_dim) {
  std::vector<BaseTensorPtr> inputs = {query,
                                       key_cache,
                                       value_cache.has_value() ? value_cache.value() : nullptr,
                                       block_tabels.has_value() ? block_tabels.value() : nullptr,
                                       context_lens.has_value() ? context_lens.value() : nullptr,
                                       antiquant_scale.has_value() ? antiquant_scale.value() : nullptr,
                                       antiquant_offset.has_value() ? antiquant_offset.value() : nullptr,
                                       attn_mask.has_value() ? attn_mask.value() : nullptr,
                                       alibi_mask.has_value() ? alibi_mask.value() : nullptr};
  std::vector<BaseTensorPtr> outputs = op->outputs();
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs);
  TransInternalShapes(&internal_outputs_shape_, outputs);

  head_num_ = static_cast<int32_t>(head_num);
  tor_ = scale_value;
  kv_head_num_ = static_cast<int32_t>(kv_head_num);
  kv_cache_quant_mode_ = static_cast<int32_t>(kv_cache_quant_mode);
  mask_mode_ = static_cast<internal::PagedAttentionParam::MaskMode>(mask_mode);
  mla_v_dim_ = static_cast<int32_t>(mla_v_dim);

  (void)GetSeqLenFromInputTensor(inputs[kIndex4], &kv_seq_len_);
  if (q_seq_lens.has_value()) {
    (void)GetSeqLenFromInputTensor(q_seq_lens.value(), &q_seq_len_);
  }

  has_attn_mask_ = attn_mask.has_value();
  has_alibi_mask_ = alibi_mask.has_value();

  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, head_num_, tor_, kv_head_num_, kv_cache_quant_mode_,
                                      mask_mode_, mla_v_dim_, q_seq_len_, kv_seq_len_, outputs);
  GetOrCreateKernel(op, inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(PagedAttention, internal::kInternalPagedAttentionOpName,
                                    InternalKernelInfoPagedAttention);
}  // namespace kernel
}  // namespace mindspore
