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

#include "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/internal/customize/paged_attention.h"
#include "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/internal/functions/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
mindspore::tensor::TensorPtr InternalPagedAttentionAscendCustomize(
  const OpPtr &op, const mindspore::tensor::TensorPtr &query_tensor,
  const mindspore::tensor::TensorPtr &key_cache_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &value_cache_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &block_tables_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &context_lens_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &antiquant_scale_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &antiquant_offset_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &attn_mask_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &q_seq_lens_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &alibi_mask_tensor, const mindspore::Int64ImmPtr &head_num,
  const mindspore::FP32ImmPtr &scale_value, const mindspore::Int64ImmPtr &kv_head_num,
  const mindspore::Int64ImmPtr &kv_cache_quant_mode, const mindspore::Int64ImmPtr &mask_mode,
  const mindspore::Int64ImmPtr &mla_v_dim) {
  op->InferOutput(query_tensor, key_cache_tensor, value_cache_tensor, block_tables_tensor, context_lens_tensor,
                  antiquant_scale_tensor, antiquant_offset_tensor, attn_mask_tensor, q_seq_lens_tensor,
                  alibi_mask_tensor, head_num, scale_value, kv_head_num, kv_cache_quant_mode, mask_mode, mla_v_dim);

  // Convert ValuePtr to c++ scalar
  auto head_num_imm = GetValue<int64_t>(head_num);
  auto scale_value_imm = GetValue<float>(scale_value);
  auto kv_head_num_imm = GetValue<int64_t>(kv_head_num);
  auto kv_cache_quant_mode_imm = GetValue<int64_t>(kv_cache_quant_mode);
  auto mask_mode_imm = GetValue<int64_t>(mask_mode);
  auto mla_v_dim_imm = GetValue<int64_t>(mla_v_dim);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query_tensor, key_cache_tensor,
                                value_cache_tensor, block_tables_tensor, antiquant_scale_tensor,
                                antiquant_offset_tensor, attn_mask_tensor, alibi_mask_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  internal_paged_attention(op, query_tensor, key_cache_tensor, value_cache_tensor, block_tables_tensor,
                           context_lens_tensor, antiquant_scale_tensor, antiquant_offset_tensor, attn_mask_tensor,
                           q_seq_lens_tensor, alibi_mask_tensor, head_num_imm, scale_value_imm, kv_head_num_imm,
                           kv_cache_quant_mode_imm, mask_mode_imm, mla_v_dim_imm);
  return op->outputs()[0];
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
