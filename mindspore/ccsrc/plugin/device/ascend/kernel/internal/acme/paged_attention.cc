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

acme::AcmeOpPtr AcmePagedAttention::CreateKernel(const acme::InputsImmutableInfoList &inputs_ii,
                                                 const acme::OutputsImmutableInfoList &outputs_ii,
                                                 const std::vector<KernelTensor *> &ms_inputs,
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
  has_attn_mask_ = (!(ms_inputs[kIndex7]->GetType()->isa<TypeNone>()));

  (void)GetSeqLenFromGraphAndCheckUpadate(kernel_name_, "q_seq_lens", &param_.q_seq_len);
  (void)GetSeqLenFromGraphAndCheckUpadate(kernel_name_, "batch_valid_length", &param_.kv_seq_len);

  CheckLookahead();
  created_flag_ = true;
  return acme::CreatePagedAttentionOp(inputs_ii, outputs_ii, param_, acme::kAcmePagedAttentionOpName);
}

bool AcmePagedAttention::UpdateParam(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  if (created_flag_) {
    // the q_seq_len and batch_valid_length are inited in CreateKernel, so there is no need to load them again
    created_flag_ = false;
    return true;
  }

  bool q_need_recreate = GetSeqLenFromGraphAndCheckUpadate(kernel_name_, "q_seq_lens", &param_.q_seq_len);
  bool kv_need_recreate = GetSeqLenFromGraphAndCheckUpadate(kernel_name_, "batch_valid_length", &param_.kv_seq_len);
  if (q_need_recreate || kv_need_recreate) {
    CheckLookahead();
    auto ret = acme_op_->UpdateParam(&param_);
    if (ret != acme::kAcmeOk) {
      MS_LOG(ERROR) << "AcmePagedAttention UpdateParam failed, kernel_name: " << kernel_name_;
      return false;
    }
    return true;
  }

  return true;
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
