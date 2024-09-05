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

#include "plugin/device/ascend/kernel/internal/acme/flash_attention_score.h"

#include <memory>
#include "kernel/kernel.h"
#include "plugin/device/ascend/kernel/internal/acme/acme_helper.h"

namespace mindspore {
namespace kernel {
acme::AcmeOpPtr AcmeFlashAttentionScore::CreateKernel(const acme::InputsImmutableInfoList &inputs_ii,
                                                      const acme::OutputsImmutableInfoList &outputs_ii,
                                                      const std::vector<KernelTensor *> &ms_inputs,
                                                      const std::vector<KernelTensor *> &ms_outputs) {
  if (ms_inputs.size() <= kIndex17) {
    MS_LOG(EXCEPTION) << "For op " << kernel_name_ << ", inputs number should be larger than " << kIndex17
                      << ", but got " << ms_inputs.size();
  }
  acme::FlashAttentionScoreParam param;
  param.mask_dtype = TransAcmeDataType(ms_inputs[kIndex6]->dtype_id());
  param.mask_dims = ms_inputs[kIndex6]->GetShapeVector();
  param.head_num = static_cast<int32_t>(ms_inputs[kIndex10]->GetValueWithCheck<int64_t>());
  param.tor = ms_inputs[kIndex12]->GetValueWithCheck<float>();
  param.pre_tokens = static_cast<int32_t>(ms_inputs[kIndex13]->GetValueWithCheck<int64_t>());
  param.next_tokens = static_cast<int32_t>(ms_inputs[kIndex14]->GetValueWithCheck<int64_t>());
  param.inner_precise = static_cast<int32_t>(ms_inputs[kIndex15]->GetValueWithCheck<int64_t>());
  param.sparse_mode = static_cast<int32_t>(ms_inputs[kIndex17]->GetValueWithCheck<int64_t>());
  return acme::CreateFlashAttentionScoreOp(inputs_ii, outputs_ii, param, acme::kAcmeFlashAttentionScoreOpName);
}

// MS_ACME_KERNEL_FACTORY_REG(FlashAttentionScore, acme::kAcmeFlashAttentionScoreOpName, AcmeFlashAttentionScore);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FlashAttentionScore, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_6);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FlashAttentionScore, OUTPUT_NUM_1, INDEX_3);
}  // namespace kernel
}  // namespace mindspore
