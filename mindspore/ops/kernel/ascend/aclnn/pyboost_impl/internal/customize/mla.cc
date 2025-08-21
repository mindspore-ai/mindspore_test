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

#include "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/internal/customize/mla.h"
#include "mindspore/ops/kernel/ascend/aclnn/pyboost_impl/internal/functions/functions.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
std::tuple<mindspore::tensor::TensorPtr, mindspore::tensor::TensorPtr> InternalMlaAscendCustomize(
  const OpPtr &op, const mindspore::tensor::TensorPtr &query_tensor, const mindspore::tensor::TensorPtr &q_rope_tensor,
  const mindspore::tensor::TensorPtr &kv_cache_tensor, const mindspore::tensor::TensorPtr &k_rope_tensor,
  const mindspore::tensor::TensorPtr &block_tables_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &attn_mask_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &deq_scale_qk_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &deq_scale_pv_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &q_seq_lens_tensor,
  const std::optional<mindspore::tensor::TensorPtr> &context_lens_tensor, const mindspore::Int64ImmPtr &head_num,
  const mindspore::FP32ImmPtr &scale_value, const mindspore::Int64ImmPtr &kv_head_num,
  const mindspore::Int64ImmPtr &mask_mode, const mindspore::Int64ImmPtr &is_ring) {
  op->InferOutput(query_tensor, q_rope_tensor, kv_cache_tensor, k_rope_tensor, block_tables_tensor, attn_mask_tensor,
                  deq_scale_qk_tensor, deq_scale_pv_tensor, q_seq_lens_tensor, context_lens_tensor, head_num,
                  scale_value, kv_head_num, mask_mode, is_ring);

  // Convert ValuePtr to c++ scalar
  auto head_num_imm = GetValue<int64_t>(head_num);
  auto scale_value_imm = GetValue<float>(scale_value);
  auto kv_head_num_imm = GetValue<int64_t>(kv_head_num);
  auto mask_mode_imm = GetValue<int64_t>(mask_mode);
  auto is_ring_imm = GetValue<int64_t>(is_ring);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query_tensor, q_rope_tensor, kv_cache_tensor,
                                k_rope_tensor, block_tables_tensor, attn_mask_tensor, deq_scale_qk_tensor,
                                deq_scale_pv_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  internal_mla(op, query_tensor, q_rope_tensor, kv_cache_tensor, k_rope_tensor, block_tables_tensor, attn_mask_tensor,
               deq_scale_qk_tensor, deq_scale_pv_tensor, q_seq_lens_tensor, context_lens_tensor, head_num_imm,
               scale_value_imm, kv_head_num_imm, mask_mode_imm, is_ring_imm);
  return std::make_tuple(op->outputs()[0], op->outputs()[1]);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
