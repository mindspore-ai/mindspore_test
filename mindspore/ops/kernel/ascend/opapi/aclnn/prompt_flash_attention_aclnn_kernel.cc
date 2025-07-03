/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/opapi/aclnn/prompt_flash_attention_aclnn_kernel.h"
#include <algorithm>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
namespace prompt_flash_attention {
void PromptFlashAttentionAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  auto actual_seq_qlen = inputs[kIndex4];
  MS_EXCEPTION_IF_NULL(actual_seq_qlen);
  std::vector<int64_t> actual_seq_qlen_array;
  if (actual_seq_qlen->type_id() != kMetaTypeNone) {
    actual_seq_qlen_array = actual_seq_qlen->GetValueWithCheck<std::vector<int64_t>>();
  }
  auto actual_seq_kvlen = inputs[kIndex5];
  MS_EXCEPTION_IF_NULL(actual_seq_kvlen);
  std::vector<int64_t> actual_seq_kvlen_array;
  if (actual_seq_kvlen->type_id() != kMetaTypeNone) {
    actual_seq_kvlen_array = actual_seq_kvlen->GetValueWithCheck<std::vector<int64_t>>();
  }
  auto num_heads = inputs[kIndex12];
  MS_EXCEPTION_IF_NULL(num_heads);
  auto num_heads_value = num_heads->GetValueWithCheck<int64_t>();

  auto scale_value = inputs[kIndex13];
  MS_EXCEPTION_IF_NULL(scale_value);
  auto scale_value_value = static_cast<double>(scale_value->GetValueWithCheck<float>());

  auto pre_tokens = inputs[kIndex14];
  MS_EXCEPTION_IF_NULL(pre_tokens);
  auto pre_tokens_value = pre_tokens->GetValueWithCheck<int64_t>();
  auto next_tokens = inputs[kIndex15];
  MS_EXCEPTION_IF_NULL(next_tokens);
  auto next_tokens_value = next_tokens->GetValueWithCheck<int64_t>();

  auto input_layout = inputs[kIndex16];
  MS_EXCEPTION_IF_NULL(input_layout);
  auto input_layout_value = input_layout->GetValueWithCheck<int64_t>();
  auto input_layout_string = FASInputLayoutMode::ConvertEnumToString(input_layout_value);

  auto num_key_value_heads = inputs[kIndex17];
  MS_EXCEPTION_IF_NULL(num_key_value_heads);
  auto num_key_value_heads_value = num_key_value_heads->GetValueWithCheck<int64_t>();

  auto sparse_mode = inputs[kIndex18];
  MS_EXCEPTION_IF_NULL(sparse_mode);
  auto sparse_mode_value = sparse_mode->GetValueWithCheck<int64_t>();

  auto inner_precise = inputs[kIndex19];
  MS_EXCEPTION_IF_NULL(inner_precise);
  auto inner_precise_value = inner_precise->GetValueWithCheck<int64_t>();

  op_type_ = "aclnnPromptFlashAttentionV3";
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex6], inputs[kIndex3],
                        actual_seq_qlen_array, actual_seq_kvlen_array, inputs[kIndex7], inputs[kIndex8],
                        inputs[kIndex9], inputs[kIndex10], inputs[kIndex11], num_heads_value, scale_value_value,
                        pre_tokens_value, next_tokens_value, input_layout_string, num_key_value_heads_value,
                        sparse_mode_value, inner_precise_value, outputs[kIndex0]);
}

bool PromptFlashAttentionAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  PFAGenerate(inputs, workspace, outputs, stream_ptr);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(PromptFlashAttention, PromptFlashAttentionAscend);
}  // namespace prompt_flash_attention
}  // namespace kernel
}  // namespace mindspore
