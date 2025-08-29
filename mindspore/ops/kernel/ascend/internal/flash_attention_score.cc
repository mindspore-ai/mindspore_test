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

#include "kernel/ascend/internal/flash_attention_score.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "utils/llm_manager.h"
#include "kernel/ascend/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
bool InternalFlashAttentionScore::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  auto &llm_manager = LLMManager::GetInstance();
  llm_manager.add_force_resize_kernel(kernel_name_);
  MS_LOG(INFO) << "Force op '" << kernel_name_ << "' to be resized to update op param 'seq_len'";
  return InternalKernelMod::Init(inputs, outputs);
}

bool InternalFlashAttentionScore::UpdateSeqLen(const std::vector<KernelTensor *> &inputs) {
  bool kv_need_recreate = false;
  if (inputs[kIndex9]->type_id() == kObjectTypeTuple) {
    kv_need_recreate = ConvertSeqLenToVectorAndCheckUpadate(inputs[kIndex9], &param_.kv_seq_len);
  } else {
    kv_need_recreate =
      GetSeqLenFromGraphAndCheckUpadate(kernel_name_, {"batch_valid_length", "actual_seq_kvlen"}, &param_.kv_seq_len);
  }

  bool q_need_recreate = kv_need_recreate;
  if (inputs[kIndex8]->type_id() == kObjectTypeTuple) {
    q_need_recreate = ConvertSeqLenToVectorAndCheckUpadate(inputs[kIndex8], &param_.q_seq_len);
  } else {
    auto &llm_manager = LLMManager::GetInstance();
    bool get_from_graph_input = false;
    for (const auto &tensor_name : {"q_seq_lens", "actual_seq_qlen"}) {
      auto seq_length_tensor = llm_manager.get_graph_input(tensor_name);
      if (seq_length_tensor != nullptr && seq_length_tensor->Size() != 0) {
        q_need_recreate = GetSeqLenFromGraphAndCheckUpadate(kernel_name_, {tensor_name}, &param_.q_seq_len);
        get_from_graph_input = true;
        break;
      }
    }
    if (!get_from_graph_input) {
      param_.q_seq_len = param_.kv_seq_len;
    }
  }

  return q_need_recreate || kv_need_recreate;
}

internal::InternalOpPtr InternalFlashAttentionScore::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                                  const internal::OutputsImmutableInfoList &outputs_ii,
                                                                  const std::vector<KernelTensor *> &ms_inputs,
                                                                  const std::vector<KernelTensor *> &ms_outputs) {
  if (ms_inputs.size() <= kIndex17) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_ << ", inputs number should be larger than " << kIndex17
                      << ", but got " << ms_inputs.size();
  }
  param_.mask_dtype = TransInternalDataType(ms_inputs[kIndex6]->dtype_id());
  param_.mask_dims = ms_inputs[kIndex6]->GetShapeVector();
  param_.head_num = static_cast<int32_t>(ms_inputs[kIndex10]->GetValueWithCheck<int64_t>());
  param_.tor = ms_inputs[kIndex12]->GetValueWithCheck<float>();
  param_.pre_tokens = static_cast<int32_t>(ms_inputs[kIndex13]->GetValueWithCheck<int64_t>());
  param_.next_tokens = static_cast<int32_t>(ms_inputs[kIndex14]->GetValueWithCheck<int64_t>());
  param_.inner_precise = static_cast<int32_t>(ms_inputs[kIndex15]->GetValueWithCheck<int64_t>());
  param_.input_layout = static_cast<int32_t>(ms_inputs[kIndex16]->GetValueWithCheck<int64_t>());
  param_.sparse_mode = static_cast<int32_t>(ms_inputs[kIndex17]->GetValueWithCheck<int64_t>());

  (void)UpdateSeqLen(ms_inputs);

  created_flag_ = true;
  return internal::CreateFlashAttentionScoreOp(inputs_ii, outputs_ii, param_,
                                               internal::kInternalFlashAttentionScoreOpName);
}

bool InternalFlashAttentionScore::UpdateParam(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  if (created_flag_) {
    // the q_seq_len and batch_valid_length are inited in CreateKernel, so there is no need to load them again
    created_flag_ = false;
    return true;
  }

  if (UpdateSeqLen(inputs)) {
    auto ret = internal_op_->UpdateParam(&param_);
    if (ret != internal::kInternalOk) {
      MS_LOG(ERROR) << "InternalFlashAttentionScore UpdateParam failed, kernel_name: " << kernel_name_;
      return false;
    }
    return true;
  }

  return true;
}

uint64_t InternalFlashAttentionScore::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return InternalTilingCache::GenerateKey(kernel_name_, inputs, param_.q_seq_len, param_.kv_seq_len);
}

MS_INTERNAL_KERNEL_FACTORY_REG(FlashAttentionScore, internal::kInternalFlashAttentionScoreOpName,
                               InternalFlashAttentionScore);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FlashAttentionScore, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_6);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FlashAttentionScore, OUTPUT_NUM_1, INDEX_3);
}  // namespace kernel
}  // namespace mindspore
