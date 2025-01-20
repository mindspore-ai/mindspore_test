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

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalKernelInfoApplyRotaryPosEmb::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                              const internal::OutputsImmutableInfoList &outputs) {
  internal::ApplyRotaryPosEmbParam param;
  param.cos_format = cos_format_;
  return internal::CreateApplyRotaryPosEmbOp(inputs, outputs, param, internal::kInternalApplyRotaryPosEmbOpName);
}

void InternalKernelInfoApplyRotaryPosEmb::Call(const std::shared_ptr<pyboost::OpRunner> &op, const ValuePtrList input_values) {
  GetInputAndOutputIndex(op, input_values);
  std::vector<BaseTensorPtr> inputs;
  std::vector<BaseTensorPtr> outputs;
  Init(input_values, inputs, outputs, op->outputs());
  auto cos_format_imm = GetValueWithCheck<int64_t>(input_values[kIndex5]);
  cos_format_ = static_cast<int32_t>(cos_format_imm);
  
  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, cos_format_);
  GetOrCreateKernel(inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(ApplyRotaryPosEmb, internal::kInternalApplyRotaryPosEmbOpName, InternalKernelInfoApplyRotaryPosEmb);
}  // namespace kernel
}  // namespace mindspore
