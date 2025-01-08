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

#include "plugin/device/ascend/kernel/internal/pyboost/apply_rotary_pos_emb.h"

#include <memory>
#include "kernel/kernel.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr AcmeKernelInfoApplyRotaryPosEmb::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                              const internal::OutputsImmutableInfoList &outputs) {
  internal::ApplyRotaryPosEmbParam param;
  param.cos_format = cos_format_;
  return internal::CreateApplyRotaryPosEmbOp(inputs, outputs, param, internal::kInternalApplyRotaryPosEmbOpName);
}

void AcmeKernelInfoApplyRotaryPosEmb::Call(const std::shared_ptr<pyboost::OpRunner> &op, const ValuePtrList input_values) {
  const auto &query_tensor = input_values[kIndex0]->cast<BaseTensorPtr>();
  const auto &key_tensor = input_values[kIndex1]->cast<BaseTensorPtr>();
  const auto &cos_tensor = input_values[kIndex2]->cast<BaseTensorPtr>();
  const auto &sin_tensor = input_values[kIndex3]->cast<BaseTensorPtr>();
  const auto &position_ids_tensor = input_values[kIndex4]->cast<BaseTensorPtr>();
  auto cos_format_imm = GetValueWithCheck<int64_t>(input_values[kIndex5]);
  cos_format_ = static_cast<int32_t>(cos_format_imm);
  
  const std::vector<BaseTensorPtr> inputs = {query_tensor, key_tensor, cos_tensor, sin_tensor, position_ids_tensor};
  auto op_key = CalcAcmeOpApiHash(kernel_name_, inputs, cos_format_);
  CallAcmeOp(op, inputs, op_key);
}
MS_ACME_KERNEL_INFO_FACTORY_REG(ApplyRotaryPosEmb, internal::kInternalApplyRotaryPosEmbOpName, AcmeKernelInfoApplyRotaryPosEmb);
}  // namespace kernel
}  // namespace mindspore
