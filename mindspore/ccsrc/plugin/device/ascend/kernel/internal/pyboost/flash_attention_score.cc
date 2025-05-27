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

#include "plugin/device/ascend/kernel/internal/pyboost/flash_attention_score.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr FlashAttentionScore::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                          const internal::OutputsImmutableInfoList &outputs) {
  internal::FlashAttentionScoreParam param;
  param.head_num = head_num_;
  param.inner_precise = inner_precise_;
  param.pre_tokens = pre_tokens_;
  param.next_tokens = next_tokens_;
  param.sparse_mode = sparse_mode_;
  param.mask_dtype = mask_dtype_;
  param.input_layout = input_layout_;
  param.mask_dims = mask_dims_;
  param.kv_seq_len = kv_seq_len_;
  param.q_seq_len = q_seq_len_;
  param.tor = tor_;
  return internal::CreateFlashAttentionScoreOp(inputs, outputs, param, internal::kInternalFlashAttentionScoreOpName);
}

uint64_t FlashAttentionScore::GenerateTilingKey(const std::string &kernel_name, const TensorPtrList &inputs) {
  return CalcInternalOpTilingHash(kernel_name, inputs, q_seq_len_, kv_seq_len_);
}

void FlashAttentionScore::Call(const std::shared_ptr<pyboost::OpRunner> &op, const TensorPtr &query,
                               const TensorPtr &key, const TensorPtr &value, const std::optional<TensorPtr> &real_shift,
                               const std::optional<TensorPtr> &drop_mask, const std::optional<TensorPtr> &padding_mask,
                               const std::optional<TensorPtr> &attn_mask, const std::vector<int64_t> &prefix,
                               const std::vector<int64_t> &actual_seq_len, const std::vector<int64_t> &actual_seq_kvlen,
                               const int64_t &head_num, const float &keep_prob, const float &scale_value,
                               const int64_t &pre_tokens, const int64_t &next_tokens, const int64_t &inner_precise,
                               const int64_t &input_layout, const int64_t &sparse_mode) {
  TensorPtrList inputs = {query, key, value, real_shift.has_value() ? real_shift.value() : nullptr,
                          attn_mask.has_value() ? attn_mask.value() : nullptr};
  TensorPtrList outputs = {op->outputs()[kIndex3]};
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs);
  TransInternalShapes(&internal_outputs_shape_, outputs);
  auto attn_mask_tensor = inputs.back();
  if (attn_mask_tensor != nullptr) {
    mask_dims_ = attn_mask_tensor->shape();
    mask_dtype_ = TransInternalDataType(attn_mask_tensor->data_type());
  }
  ConvertVectorDtype<int32_t, int64_t>(&q_seq_len_, actual_seq_len);
  ConvertVectorDtype<int32_t, int64_t>(&kv_seq_len_, actual_seq_kvlen);

  head_num_ = static_cast<int32_t>(head_num);
  tor_ = scale_value;
  pre_tokens_ = static_cast<int32_t>(pre_tokens);
  next_tokens_ = static_cast<int32_t>(next_tokens);
  inner_precise_ = static_cast<int32_t>(inner_precise);
  input_layout_ = static_cast<int32_t>(input_layout);
  sparse_mode_ = static_cast<int32_t>(sparse_mode);

  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, head_num_, tor_, pre_tokens_, next_tokens_, inner_precise_,
                                      input_layout_, sparse_mode_, mask_dims_, q_seq_len_, kv_seq_len_, outputs);
  GetOrCreateKernel(op, inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(FlashAttentionScore, internal::kInternalFlashAttentionScoreOpName,
                                    FlashAttentionScore);
}  // namespace kernel
}  // namespace mindspore
