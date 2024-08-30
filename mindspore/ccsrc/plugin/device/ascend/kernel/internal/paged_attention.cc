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

#include "plugin/device/ascend/kernel/internal/paged_attention.h"

#include <memory>
#include <string>

#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "utils/llm_manager.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace kernel {
bool InternalPagedAttention::Init(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto &enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  enable_custom_pa_ = (std::find(enable_op_list.begin(), enable_op_list.end(), kernel_name_) != enable_op_list.end());
  auto &llm_manager = LLMManager::GetInstance();
  llm_manager.add_force_resize_kernel(kernel_name_);
  MS_LOG(INFO) << "Force op '" << kernel_name_ << "' to be resized to update op param 'seq_len'";
  return InternalKernelMod::Init(inputs, outputs);
}

internal::OpParamPtr InternalPagedAttention::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                           const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::PagedAttention;
  internal::MixParam op_param;
  op_param.maskType = internal::MixParam::MaskType::MASK_TYPE_NONE;

  if (soc_ == "ascend310p") {
    op_param.mixType = internal::MixParam::MixType::MIX_PAGED_ATTENTION_NZ_MASK;
  } else {
    op_param.mixType = internal::MixParam::MixType::MIX_PAGED_ATTENTION_MASK_ND;
  }

  op_param.headSize = static_cast<int32_t>(inputs[kIndex9]->GetValueWithCheck<int64_t>());
  op_param.tor = inputs[kIndex10]->GetValueWithCheck<float>();
  op_param.kvHead = static_cast<int32_t>(inputs[kIndex11]->GetValueWithCheck<int64_t>());

  if (!enable_custom_pa_) {
    GetSeqLenFromGraphInputOrEnv(kernel_name_, "batch_valid_length", "MS_INTERNAL_KV_SEQ_LEN", &kv_seq_len_);
    for (const auto &item : kv_seq_len_) {
      (void)op_param.kvSeqLen.emplace_back(item);
    }
  }

  GetSeqLenFromGraphInputOrEnv(kernel_name_, "q_seq_lens", "MS_INTERNAL_Q_SEQ_LEN", &q_seq_len_);
  bool no_need_lookahead =
    std::all_of(q_seq_len_.begin(), q_seq_len_.end(), [](int32_t seq_len) { return seq_len == 1; });
  if (!no_need_lookahead) {
    for (const auto &item : q_seq_len_) {
      (void)op_param.qSeqLen.emplace_back(item);
    }
    // input attn_mask is not None
    if (!(inputs[kIndex7]->GetType()->isa<TypeNone>())) {
      op_param.maskType = internal::MixParam::MaskType::MASK_TYPE_LOOK_AHEAD;
    }
  }

  param_ptr->specificParam = op_param;
  return param_ptr;
}

uint64_t InternalPagedAttention::GenTilingCacheKey(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return TilingCacheMgr::GetInstance().GenTilingCacheKey(kernel_name_, inputs, q_seq_len_, kv_seq_len_);
}

MS_INTERNAL_KERNEL_FACTORY_REG(PagedAttention, InternalPagedAttention);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(PagedAttention, INPUT_NUM_8, INDEX_0, INDEX_1, INDEX_2, INDEX_4, INDEX_3, INDEX_5,
                                     INDEX_6, INDEX_7);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(PagedAttention, OUTPUT_NUM_1, INDEX_0);
}  // namespace kernel
}  // namespace mindspore
