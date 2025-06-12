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

#include "plugin/device/ascend/kernel/internal/pyboost/apply_rotary_pos_emb.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr ApplyRotaryPosEmb::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                        const internal::OutputsImmutableInfoList &outputs) {
  internal::ApplyRotaryPosEmbParam param;
  param.cos_format = cos_format_;
  return internal::CreateApplyRotaryPosEmbOp(inputs, outputs, param, internal::kInternalApplyRotaryPosEmbOpName);
}

void ApplyRotaryPosEmb::Call(const std::shared_ptr<pyboost::OpRunner> &op, const TensorPtr &query_tensor,
                             const TensorPtr &key_tensor, const TensorPtr &cos_tensor, const TensorPtr &sin_tensor,
                             const TensorPtr &position_ids_tensor, const int64_t &cos_format) {
  TensorPtrList inputs = {query_tensor, key_tensor, cos_tensor, sin_tensor, position_ids_tensor};
  TensorPtrList outputs = op->outputs();
  internal_inputs_shape_.resize(inputs.size());
  internal_outputs_shape_.resize(outputs.size());
  TransInternalShapes(&internal_inputs_shape_, inputs);
  TransInternalShapes(&internal_outputs_shape_, outputs);
  cos_format_ = cos_format;
  auto op_key = CalcInternalOpApiHash(kernel_name_, inputs, cos_format_, outputs);
  GetOrCreateKernel(op, inputs, outputs, op_key);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(ApplyRotaryPosEmb, ApplyRotaryPosEmb);
}  // namespace kernel
}  // namespace mindspore
