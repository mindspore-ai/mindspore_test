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

#include "kernel/ascend/internal/mla.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/internal/internal_kernel_in_out_map.h"
#include "utils/llm_manager.h"
#include "kernel/ascend/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
bool InternalMla::Init(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto &llm_manager = LLMManager::GetInstance();
  llm_manager.add_force_resize_kernel(kernel_name_);
  MS_LOG(INFO) << "Force op '" << kernel_name_ << "' to be resized to update op param 'seq_len'";
  return InternalKernelMod::Init(inputs, outputs);
}

internal::InternalOpPtr InternalMla::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                  const internal::OutputsImmutableInfoList &outputs_ii,
                                                  const std::vector<KernelTensor *> &ms_inputs,
                                                  const std::vector<KernelTensor *> &ms_outputs) {
  param_.type = internal::MLAParam::kSplitCache;
  param_.head_size = static_cast<int32_t>(ms_inputs[kIndex10]->GetValueWithCheck<int64_t>());
  param_.tor = ms_inputs[kIndex11]->GetValueWithCheck<float>();
  param_.kv_head = static_cast<int32_t>(ms_inputs[kIndex12]->GetValueWithCheck<int64_t>());
  param_.mask_type = static_cast<internal::MLAParam::MaskType>(ms_inputs[kIndex13]->GetValueWithCheck<int64_t>());
  param_.is_ring = static_cast<int32_t>(ms_inputs[kIndex14]->GetValueWithCheck<int64_t>());

  (void)GetSeqLenFromGraphAndCheckUpadate(kernel_name_, {"q_seq_lens"}, &param_.q_seq_len);
  (void)GetSeqLenFromGraphAndCheckUpadate(kernel_name_, {"batch_valid_length"}, &param_.kv_seq_len);

  created_flag_ = true;
  return internal::CreateMLAOp(inputs_ii, outputs_ii, param_, internal::kInternalMLAOpName);
}

bool InternalMla::UpdateParam(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  if (created_flag_) {
    // the q_seq_len and batch_valid_length are inited in CreateKernel, so there is no need to load them again
    created_flag_ = false;
    return true;
  }

  bool q_need_recreate = GetSeqLenFromGraphAndCheckUpadate(kernel_name_, {"q_seq_lens"}, &param_.q_seq_len);
  bool kv_need_recreate = GetSeqLenFromGraphAndCheckUpadate(kernel_name_, {"batch_valid_length"}, &param_.kv_seq_len);
  if (q_need_recreate || kv_need_recreate) {
    auto ret = internal_op_->UpdateParam(&param_);
    if (ret != internal::kInternalOk) {
      MS_LOG(ERROR) << "InternalMla UpdateParam failed, kernel_name: " << kernel_name_;
      return false;
    }
    return true;
  }

  return true;
}

uint64_t InternalMla::GenerateTilingKey(const std::vector<KernelTensor *> &inputs) {
  // User defined CacheKey, the inputs should include all the factors which will affect tiling result.
  return InternalTilingCache::GenerateKey(kernel_name_, inputs, param_.q_seq_len, param_.kv_seq_len);
}

MS_INTERNAL_KERNEL_FACTORY_REG(Mla, internal::kInternalMLAOpName, InternalMla);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(Mla, INPUT_NUM_8, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4, INDEX_5, INDEX_6,
                                     INDEX_7);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(Mla, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
