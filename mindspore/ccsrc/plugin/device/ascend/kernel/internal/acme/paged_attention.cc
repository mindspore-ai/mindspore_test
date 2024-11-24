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

#include "plugin/device/ascend/kernel/internal/acme/paged_attention.h"

#include <memory>
#include "kernel/kernel.h"
#include "utils/llm_manager.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
bool AcmePagedAttention::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto &llm_manager = LLMManager::GetInstance();
  llm_manager.add_force_resize_kernel(kernel_name_);
  MS_LOG(INFO) << "Force op '" << kernel_name_ << "' to be resized to update op param 'seq_len'";
  return AcmeKernelMod::Init(inputs, outputs);
}

void AcmePagedAttention::CreateOpParam(const std::vector<KernelTensor *> &ms_inputs,
                                       const std::vector<KernelTensor *> &ms_outputs) {
  auto last_input_index = kIndex12;
  if (ms_inputs.size() <= last_input_index) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_ << ", inputs number should be larger than " << last_input_index
                      << ", but got " << ms_inputs.size();
  }
  param_.head_num = static_cast<int32_t>(ms_inputs[kIndex9]->GetValueWithCheck<int64_t>());
  param_.tor = ms_inputs[kIndex10]->GetValueWithCheck<float>();
  param_.kv_head_num = static_cast<int32_t>(ms_inputs[kIndex11]->GetValueWithCheck<int64_t>());
  param_.kv_cache_quant_mode = ms_inputs[last_input_index]->GetValueWithCheck<int64_t>();

  if (!init_) {
    MS_LOG(INFO) << "PagedAttention: Update of q_seq_len & kv_seq_len is skipped here as they have been updated in "
                    "IsParamChanged function after initialized";
    (void)GetSeqLenFromGraphAndCheckUpadate(kernel_name_, "q_seq_lens", &param_.q_seq_len);
    (void)GetSeqLenFromGraphAndCheckUpadate(kernel_name_, "batch_valid_length", &param_.kv_seq_len);
    init_ = true;
  }
  param_.mask_type = acme::PagedAttentionParam::MaskType::kMaskTypeNone;
  bool enable_lookahead =
    std::any_of(param_.q_seq_len.begin(), param_.q_seq_len.end(), [](int32_t seq_len) { return seq_len > 1; });
  bool has_attn_mask = (!(ms_inputs[kIndex7]->GetType()->isa<TypeNone>()));

  if (enable_lookahead) {
    if (has_attn_mask) {
      param_.mask_type = acme::PagedAttentionParam::MaskType::kMaskTypeLookAhead;
    }
  } else {
    param_.q_seq_len.clear();
  }
}

acme::AcmeOpPtr AcmePagedAttention::CreateKernel(const acme::InputsImmutableInfoList &inputs_ii,
                                                 const acme::OutputsImmutableInfoList &outputs_ii,
                                                 const std::vector<KernelTensor *> &ms_inputs,
                                                 const std::vector<KernelTensor *> &ms_outputs) {
  CreateOpParam(ms_inputs, ms_outputs);
  return acme::CreatePagedAttentionOp(inputs_ii, outputs_ii, param_, acme::kAcmePagedAttentionOpName);
}

bool AcmePagedAttention::IsParamChanged() {
  bool q_need_recreate = GetSeqLenFromGraphAndCheckUpadate(kernel_name_, "q_seq_lens", &param_.q_seq_len);
  bool kv_need_recreate = GetSeqLenFromGraphAndCheckUpadate(kernel_name_, "batch_valid_length", &param_.kv_seq_len);
  if (q_need_recreate || kv_need_recreate) {
    return true;
  }
  return false;
}

uint64_t AcmePagedAttention::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return AcmeTilingCache::GenerateKey(kernel_name_, inputs, param_.q_seq_len, param_.kv_seq_len);
}

MS_ACME_KERNEL_FACTORY_REG(PagedAttention, acme::kAcmePagedAttentionOpName, AcmePagedAttention);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(PagedAttention, INPUT_NUM_8, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4, INDEX_5,
                                     INDEX_6, INDEX_7);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(PagedAttention, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
