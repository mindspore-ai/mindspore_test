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

#include "kernel/ascend/internal/pyboost/flash_attention_score.h"

#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr FlashAttentionScore::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                          const internal::OutputsImmutableInfoList &outputs) {
  created_flag_ = true;
  return internal::CreateFlashAttentionScoreOp(inputs, outputs, param_, internal::kInternalFlashAttentionScoreOpName);
}

bool FlashAttentionScore::UpdateParam() {
  if (created_flag_) {
    created_flag_ = false;
    return true;
  }

  auto ret = internal_op_->UpdateParam(&param_);
  if (ret != internal::kInternalOk) {
    MS_LOG(ERROR) << "InternalFlashAttentionScore UpdateParam failed, kernel_name: " << kernel_name_;
    return false;
  }
  return true;
}

uint64_t FlashAttentionScore::GetOrGenerateOpTilingKey(const uint64_t &tiling_key) const {
  return CalcInternalOpTilingHash(kernel_name_, tiling_key, param_.q_seq_len, param_.kv_seq_len);
}

void FlashAttentionScore::Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key,
                               const uint64_t &tiling_key, const TensorPtr &query, const TensorPtr &key,
                               const TensorPtr &value, const std::optional<TensorPtr> &real_shift,
                               const std::optional<TensorPtr> &drop_mask, const std::optional<TensorPtr> &padding_mask,
                               const std::optional<TensorPtr> &attn_mask, const std::vector<int64_t> &prefix,
                               const std::vector<int64_t> &actual_seq_len, const std::vector<int64_t> &actual_seq_kvlen,
                               const int64_t &head_num, const float &keep_prob, const float &scale_value,
                               const int64_t &pre_tokens, const int64_t &next_tokens, const int64_t &inner_precise,
                               const int64_t &input_layout, const int64_t &sparse_mode) {
  TensorPtrList inputs = {query, key, value, real_shift.has_value() ? real_shift.value() : nullptr,
                          attn_mask.has_value() ? attn_mask.value() : nullptr};
  TensorPtrList outputs = {op->outputs()[kIndex3]};
  TransInternalShapes(inputs, outputs);
  auto attn_mask_tensor = inputs.back();
  if (attn_mask_tensor != nullptr) {
    param_.mask_dims = attn_mask_tensor->shape();
    param_.mask_dtype = TransInternalDataType(attn_mask_tensor->data_type());
  }
  ConvertVectorDtype<int32_t, int64_t>(&param_.q_seq_len, actual_seq_len);
  ConvertVectorDtype<int32_t, int64_t>(&param_.kv_seq_len, actual_seq_kvlen);

  param_.head_num = static_cast<int32_t>(head_num);
  param_.tor = scale_value;
  param_.pre_tokens = static_cast<int32_t>(pre_tokens);
  param_.next_tokens = static_cast<int32_t>(next_tokens);
  param_.inner_precise = static_cast<int32_t>(inner_precise);
  param_.input_layout = static_cast<int32_t>(input_layout);
  param_.sparse_mode = static_cast<int32_t>(sparse_mode);

  GetOrCreateKernel(op, op_key, tiling_key, inputs, outputs);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(FlashAttentionScore, FlashAttentionScore);
}  // namespace kernel
}  // namespace mindspore
