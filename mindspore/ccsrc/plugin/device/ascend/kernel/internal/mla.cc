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

#include "plugin/device/ascend/kernel/internal/mla.h"

#include <memory>
#include "common/kernel.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "utils/llm_manager.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalMla::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                  const internal::OutputsImmutableInfoList &outputs_ii,
                                                  const std::vector<KernelTensor *> &ms_inputs,
                                                  const std::vector<KernelTensor *> &ms_outputs) {
  internal::MLAParam param;
  param.type = internal::MLAParam::kSplitCache;
  param.head_size = static_cast<int32_t>(ms_inputs[kIndex10]->GetValueWithCheck<int64_t>());
  param.tor = ms_inputs[kIndex11]->GetValueWithCheck<float>();
  param.kv_head = static_cast<int32_t>(ms_inputs[kIndex12]->GetValueWithCheck<int64_t>());
  param.mask_type = static_cast<internal::MLAParam::MaskType>(ms_inputs[kIndex13]->GetValueWithCheck<int64_t>());
  param.is_ring = static_cast<int32_t>(ms_inputs[kIndex14]->GetValueWithCheck<int64_t>());

  (void)GetSeqLenFromGraphAndCheckUpadate(kernel_name_, {"q_seq_lens"}, &param.q_seq_len);
  (void)GetSeqLenFromGraphAndCheckUpadate(kernel_name_, {"batch_valid_length"}, &param.kv_seq_len);

  return internal::CreateMLAOp(inputs_ii, outputs_ii, param, internal::kInternalMLAOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(Mla, internal::kInternalMLAOpName, InternalMla);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(Mla, INPUT_NUM_8, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4, INDEX_5, INDEX_6,
                                     INDEX_7);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(Mla, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
