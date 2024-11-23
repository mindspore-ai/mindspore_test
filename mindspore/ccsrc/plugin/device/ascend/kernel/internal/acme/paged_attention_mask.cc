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

#include "plugin/device/ascend/kernel/internal/acme/paged_attention_mask.h"

#include <memory>
#include "kernel/kernel.h"
#include "utils/llm_manager.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
bool AcmePagedAttentionMask::Init(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  auto &llm_manager = LLMManager::GetInstance();
  llm_manager.add_force_resize_kernel(kernel_name_);
  MS_LOG(INFO) << "Force op '" << kernel_name_ << "' to be resized to update op param 'seq_len'";
  return AcmeKernelMod::Init(inputs, outputs);
}

acme::AcmeOpPtr AcmePagedAttentionMask::CreateKernel(const acme::InputsImmutableInfoList &inputs_ii,
                                                     const acme::OutputsImmutableInfoList &outputs_ii,
                                                     const std::vector<KernelTensor *> &ms_inputs,
                                                     const std::vector<KernelTensor *> &ms_outputs) {
  param_.head_num = static_cast<int32_t>(ms_inputs[kIndex8]->GetValueWithCheck<int64_t>());
  param_.tor = ms_inputs[kIndex9]->GetValueWithCheck<float>();
  param_.kv_head_num = static_cast<int32_t>(ms_inputs[kIndex10]->GetValueWithCheck<int64_t>());
  param_.mask_type = acme::PagedAttentionParam::MaskType::kMaskTypeNone;

  // input alibi_mask is not None
  if (!(ms_inputs[kIndex7]->GetType()->isa<TypeNone>())) {
    param_.mask_type = acme::PagedAttentionParam::MaskType::kMaskTypeAlibi;
  }

  auto kv_cache_quant_mode = ms_inputs[kIndex11]->GetValueWithCheck<int64_t>();
  param_.kv_cache_quant_mode = kv_cache_quant_mode;

  return acme::CreatePagedAttentionOp(inputs_ii, outputs_ii, param_, acme::kAcmePagedAttentionOpName);
}

bool AcmePagedAttentionMask::IsNeedRecreate(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  bool q_need_recreate = GetSeqLenFromGraphAndCheckUpadate(kernel_name_, "q_seq_lens", &param_.q_seq_len);
  bool kv_need_recreate = GetSeqLenFromGraphAndCheckUpadate(kernel_name_, "batch_valid_length", &param_.kv_seq_len);
  if (q_need_recreate || kv_need_recreate) {
    return true;
  }
  return AcmeKernelMod::IsNeedRecreate(inputs, outputs);
}

uint64_t AcmePagedAttentionMask::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return AcmeTilingCache::GenerateKey(kernel_name_, inputs, param_.q_seq_len, param_.kv_seq_len);
}

MS_ACME_KERNEL_FACTORY_REG(PagedAttentionMask, acme::kAcmePagedAttentionOpName, AcmePagedAttentionMask);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(PagedAttentionMask, INPUT_NUM_8, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4,
                                     INDEX_5, INDEX_6, INDEX_7);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(PagedAttentionMask, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
