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
#include "utils/ms_context.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
bool AcmePagedAttention::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto &llm_manager = LLMManager::GetInstance();
  llm_manager.add_force_resize_kernel(kernel_name_);
  MS_LOG(INFO) << "Force op '" << kernel_name_ << "' to be resized to update op param 'seq_len'";
  return AcmeKernelMod::Init(inputs, outputs);
}

acme::AcmeOpPtr AcmePagedAttention::CreateKernel(const acme::InputsImmutableInfoList &inputs_ii,
                                                 const acme::OutputsImmutableInfoList &outputs_ii,
                                                 const std::vector<KernelTensor *> &ms_inputs,
                                                 const std::vector<KernelTensor *> &ms_outputs) {
  if (ms_inputs.size() <= kIndex11) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_ << ", inputs number should be larger than " << kIndex9
                      << ", but got " << ms_inputs.size();
  }
  acme::PagedAttentionParam param;
  param.head_num = static_cast<int32_t>(ms_inputs[kIndex9]->GetValueWithCheck<int64_t>());
  param.tor = ms_inputs[kIndex10]->GetValueWithCheck<float>();
  param.kv_head_num = static_cast<int32_t>(ms_inputs[kIndex11]->GetValueWithCheck<int64_t>());
  param.mask_type = acme::PagedAttentionParam::MaskType::kMaskTypeNone;
  GetSeqLenFromGraphInputOrEnv(kernel_name_, "batch_valid_length", "MS_INTERNAL_KV_SEQ_LEN", &kv_seq_len_);
  for (const auto &item : kv_seq_len_) {
    (void)param.kv_seq_len.emplace_back(item);
  }

  GetSeqLenFromGraphInputOrEnv(kernel_name_, "q_seq_lens", "MS_INTERNAL_Q_SEQ_LEN", &q_seq_len_);
  bool no_need_lookahead =
    std::all_of(q_seq_len_.begin(), q_seq_len_.end(), [](int32_t seq_len) { return seq_len == 1; });
  if (!no_need_lookahead) {
    for (const auto &item : q_seq_len_) {
      (void)param.q_seq_len.emplace_back(item);
    }
    // input attn_mask is not None
    if (!(ms_inputs[kIndex7]->GetType()->isa<TypeNone>())) {
      param.mask_type = acme::PagedAttentionParam::MaskType::kMaskTypeLookAhead;
    }
  }
  auto kv_cache_quant_mode = ms_inputs[kIndex12]->GetValueWithCheck<int64_t>();
  param.kv_cache_quant_mode = kv_cache_quant_mode;
  return acme::CreatePagedAttentionOp(inputs_ii, outputs_ii, param, acme::kAcmePagedAttentionOpName);
}

bool AcmePagedAttention::IsNeedRecreate(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  // (todo) if q_seq_len_ or kv_seq_len_ changed , need to recreate
  return true;
}

uint64_t AcmePagedAttention::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return AcmeTilingCache::GenerateKey(kernel_name_, inputs, q_seq_len_, kv_seq_len_);
}

MS_ACME_KERNEL_FACTORY_REG(PagedAttention, acme::kAcmePagedAttentionOpName, AcmePagedAttention);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(PagedAttention, INPUT_NUM_8, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4, INDEX_5,
                                     INDEX_6, INDEX_7);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(PagedAttention, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
