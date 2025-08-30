/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/incre_flash_attention_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "plugin/ascend/res_manager/op_adapter/op_adapter_base.h"
#include "utils/llm_manager.h"

namespace mindspore {
namespace kernel {
namespace incre_flash_attention {

std::vector<int64_t> ConvertActualSeqLengthsToVector(KernelTensor *const actual_seq_length_ptr) {
  MS_EXCEPTION_IF_NULL(actual_seq_length_ptr);
  std::vector<int64_t> actual_seq_lengths_vector;
  if (actual_seq_length_ptr->type_id() != kMetaTypeNone) {
    TypeId actual_seq_lengths_dtype_id = actual_seq_length_ptr->dtype_id();
    if (actual_seq_lengths_dtype_id == kNumberTypeInt64) {
      actual_seq_lengths_vector = actual_seq_length_ptr->GetValueWithCheck<std::vector<int64_t>>();
    } else if (actual_seq_lengths_dtype_id == kNumberTypeInt32) {
      std::vector<int32_t> actual_seq_lengths_vector_int32 =
        actual_seq_length_ptr->GetValueWithCheck<std::vector<int32_t>>();
      actual_seq_lengths_vector.assign(actual_seq_lengths_vector_int32.begin(), actual_seq_lengths_vector_int32.end());
    } else {
      MS_LOG(EXCEPTION) << "actual_seq_lengths data type must be Int32 or Int64, but got "
                        << TypeIdToString(actual_seq_lengths_dtype_id);
    }
  }
  return actual_seq_lengths_vector;
}

bool IncreFlashAttentionAscend::Init(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  auto &llm_manager = LLMManager::GetInstance();
  llm_manager.add_force_resize_kernel(kernel_name_);
  return true;
}

void IncreFlashAttentionAscend::SetScalarParam(const std::vector<KernelTensor *> &inputs) {
  num_heads_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex15]);
  auto input_layout = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex16]);
  input_layout_str_ = device::ascend::FASInputLayoutMode::ConvertEnumToString(input_layout);

  scale_value_d_ = static_cast<double>(device::ascend::ConvertKernelTensor<float>(inputs[kIndex17]));
  num_key_value_heads_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex18]);

  block_size_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex19]);
  inner_precise_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex20]);
  return;
}

void IncreFlashAttentionAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(inputs[kIndex1]);
  key_vector_ = {inputs[kIndex1]};
  MS_EXCEPTION_IF_NULL(inputs[kIndex2]);
  value_vector_ = {inputs[kIndex2]};

  SetScalarParam(inputs);
  std::vector<int64_t> actual_kv_lengths_vector = {};
  auto ret =
    LLMManager::GetInstance().GetGraphInputToVector<int32_t, int64_t>("batch_valid_length", &actual_kv_lengths_vector);
  if (!ret) {
    actual_kv_lengths_vector = ConvertActualSeqLengthsToVector(inputs[kIndex4]);
  }
  actual_kv_lengths_vector_pair_ = std::make_pair(actual_kv_lengths_vector, true);

  // For interface aclnnIncreFlashAttentionV4, param inputs[kIndex5] (pse_shift) should follow param value_vector
  GetWorkspaceForResize(inputs[kIndex0], key_vector_, value_vector_, inputs[kIndex5], inputs[kIndex3],
                        actual_kv_lengths_vector_pair_, inputs[kIndex6], inputs[kIndex7], inputs[kIndex8],
                        inputs[kIndex9], inputs[kIndex10], inputs[kIndex11], inputs[kIndex12], inputs[kIndex13],
                        inputs[kIndex14], num_heads_, scale_value_d_, input_layout_str_, num_key_value_heads_,
                        block_size_, inner_precise_, outputs[kIndex0]);
}

bool IncreFlashAttentionAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  // For interface aclnnIncreFlashAttentionV4, param inputs[kIndex5] (pse_shift) should follow param value_vector
  RunOp(stream_ptr, workspace, inputs[kIndex0], key_vector_, value_vector_, inputs[kIndex5], inputs[kIndex3],
        actual_kv_lengths_vector_pair_, inputs[kIndex6], inputs[kIndex7], inputs[kIndex8], inputs[kIndex9],
        inputs[kIndex10], inputs[kIndex11], inputs[kIndex12], inputs[kIndex13], inputs[kIndex14], num_heads_,
        scale_value_d_, input_layout_str_, num_key_value_heads_, block_size_, inner_precise_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(IncreFlashAttention, IncreFlashAttentionAscend);
}  // namespace incre_flash_attention
}  // namespace kernel
}  // namespace mindspore
