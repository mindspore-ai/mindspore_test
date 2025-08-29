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

#include "kernel/ascend/internal/apply_rotary_pos_emb.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalApplyRotaryPosEmb::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                                const internal::OutputsImmutableInfoList &outputs_ii,
                                                                const std::vector<KernelTensor *> &ms_inputs,
                                                                const std::vector<KernelTensor *> &ms_outputs) {
  internal::ApplyRotaryPosEmbParam param;
  auto last_input = ms_inputs.at(kIndex5);
  if (last_input->dtype_id() == TypeId::kNumberTypeInt64) {
    param.cos_format = static_cast<int32_t>(last_input->GetValue<int64_t>().value());
  } else {
    MS_LOG(EXCEPTION) << "ApplyRotaryPosEmb input[5] dtype is not kNumberTypeInt64";
  }
  return internal::CreateApplyRotaryPosEmbOp(inputs_ii, outputs_ii, param, internal::kInternalApplyRotaryPosEmbOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(ApplyRotaryPosEmb, internal::kInternalApplyRotaryPosEmbOpName,
                               InternalApplyRotaryPosEmb);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(ApplyRotaryPosEmb, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_4);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(ApplyRotaryPosEmb, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
