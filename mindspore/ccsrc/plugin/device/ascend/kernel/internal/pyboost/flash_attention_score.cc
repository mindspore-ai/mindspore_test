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

#include <memory>
#include <vector>
#include "kernel/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalKernelInfoFlashAttentionScore::CreateKernel(
  const internal::InputsImmutableInfoList &inputs, const internal::OutputsImmutableInfoList &outputs) {
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

void InternalKernelInfoFlashAttentionScore::Call(const std::shared_ptr<pyboost::OpRunner> &op,
                                                 const ValuePtrList input_values) {
  auto device_ctx = op->device_context();
  device_ctx->device_res_manager_->BindDeviceToCurrentThread(false);
  GetInputAndOutputIndex(op, input_values);
  std::vector<BaseTensorPtr> inputs;
  std::vector<BaseTensorPtr> outputs;
  Init(input_values, inputs, outputs, op->outputs());

  if (!input_values[kIndex6]->isa<None>()) {
    auto attn_mask_tensor = input_values[kIndex6]->cast<BaseTensorPtr>();
    mask_dims_ = attn_mask_tensor->shape();
    mask_dtype_ = TransInternalDataType(attn_mask_tensor->data_type());
  }
  if (!input_values[kIndex8]->isa<None>()) {
    const auto &actual_seq_qlen_vec = GetValueWithCheck<std::vector<int64_t>>(input_values[kIndex8]);
    ConvertVectorDtype<int32_t, int64_t>(q_seq_len_, actual_seq_qlen_vec);
  }
  if (!input_values[kIndex9]->isa<None>()) {
    const auto &actual_seq_kvlen_vec = GetValueWithCheck<std::vector<int64_t>>(input_values[kIndex9]);
    ConvertVectorDtype<int32_t, int64_t>(kv_seq_len_, actual_seq_kvlen_vec);
  }

  head_num_ = static_cast<int32_t>(GetValueWithCheck<int64_t>(input_values[kIndex10]));
  tor_ = GetValueWithCheck<float>(input_values[kIndex12]);
  pre_tokens_ = static_cast<int32_t>(GetValueWithCheck<int64_t>(input_values[kIndex13]));
  next_tokens_ = static_cast<int32_t>(GetValueWithCheck<int64_t>(input_values[kIndex14]));
  inner_precise_ = static_cast<int32_t>(GetValueWithCheck<int64_t>(input_values[kIndex15]));
  input_layout_ = static_cast<int32_t>(GetValueWithCheck<int64_t>(input_values[kIndex16]));
  sparse_mode_ = static_cast<int32_t>(GetValueWithCheck<int64_t>(input_values[kIndex17]));

  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, head_num_, tor_, pre_tokens_, next_tokens_, inner_precise_,
                                      input_layout_, sparse_mode_, mask_dims_, q_seq_len_, kv_seq_len_);
  GetOrCreateKernel(inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(FlashAttentionScore, internal::kInternalFlashAttentionScoreOpName,
                                    InternalKernelInfoFlashAttentionScore);
}  // namespace kernel
}  // namespace mindspore
