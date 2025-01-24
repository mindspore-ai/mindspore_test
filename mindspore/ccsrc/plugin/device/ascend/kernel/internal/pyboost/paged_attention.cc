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
#include <vector>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalKernelInfoPagedAttention::CreateKernel(
  const internal::InputsImmutableInfoList &inputs, const internal::OutputsImmutableInfoList &outputs) {
  internal::PagedAttentionParam param;
  param.head_num = head_num_;
  param.kv_head_num = kv_head_num_;
  param.tor = tor_;
  param.kv_cache_quant_mode = kv_cache_quant_mode_;

  auto enable_lookahead =
    std::any_of(q_seq_len_.begin(), q_seq_len_.end(), [](int32_t seq_len) { return seq_len > 1; });
  if (enable_lookahead) {
    if (has_attn_mask_) {
      param.mask_type = internal::PagedAttentionParam::MaskType::kMaskTypeLookAhead;
    }
  } else {
    q_seq_len_.clear();
  }

  param.q_seq_len = q_seq_len_;
  param.kv_seq_len = kv_seq_len_;

  return internal::CreatePagedAttentionOp(inputs, outputs, param, internal::kInternalPagedAttentionOpName);
}

void InternalKernelInfoPagedAttention::Call(const std::shared_ptr<pyboost::OpRunner> &op,
                                            const ValuePtrList input_values) {
  auto device_ctx = op->device_context();
  device_ctx->device_res_manager_->BindDeviceToCurrentThread(false);
  GetInputAndOutputIndex(op, input_values);
  std::vector<BaseTensorPtr> inputs;
  std::vector<BaseTensorPtr> outputs;
  Init(input_values, inputs, outputs, op->outputs());

  head_num_ = static_cast<int32_t>(GetValueWithCheck<int64_t>(input_values[kIndex9]));
  tor_ = GetValueWithCheck<float>(input_values[kIndex10]);
  kv_head_num_ = static_cast<int32_t>(GetValueWithCheck<int64_t>(input_values[kIndex11]));
  kv_cache_quant_mode_ = static_cast<int32_t>(GetValueWithCheck<int64_t>(input_values[kIndex12]));

  (void)GetSeqLenFromInputTensor(inputs[kIndex4], &kv_seq_len_);
  if (!input_values[kIndex8]->isa<None>()) {
    auto q_seq_lens = input_values[kIndex8]->cast<BaseTensorPtr>();
    (void)GetSeqLenFromInputTensor(q_seq_lens, &q_seq_len_);
  }

  has_attn_mask_ = inputs.back() != nullptr;

  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, head_num_, tor_, kv_head_num_, kv_cache_quant_mode_);
  GetOrCreateKernel(inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(PagedAttention, internal::kInternalPagedAttentionOpName,
                                    InternalKernelInfoPagedAttention);
}  // namespace kernel
}  // namespace mindspore
