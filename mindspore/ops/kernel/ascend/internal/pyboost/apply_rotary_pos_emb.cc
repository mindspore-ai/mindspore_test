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

#include "kernel/ascend/internal/pyboost/apply_rotary_pos_emb.h"

#include "include/runtime/hardware_abstract/kernel_base/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr ApplyRotaryPosEmb::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                        const internal::OutputsImmutableInfoList &outputs) {
  return internal::CreateApplyRotaryPosEmbOp(inputs, outputs, param_, internal::kInternalApplyRotaryPosEmbOpName);
}

void ApplyRotaryPosEmb::Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key,
                             const uint64_t &tiling_key, const TensorPtr &query_tensor, const TensorPtr &key_tensor,
                             const TensorPtr &cos_tensor, const TensorPtr &sin_tensor,
                             const TensorPtr &position_ids_tensor, const int64_t &cos_format) {
  TensorPtrList inputs = {query_tensor, key_tensor, cos_tensor, sin_tensor, position_ids_tensor};
  TensorPtrList outputs = op->outputs();
  TransInternalShapes(inputs, outputs);
  param_.cos_format = cos_format;
  GetOrCreateKernel(op, op_key, tiling_key, inputs, outputs);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(ApplyRotaryPosEmb, ApplyRotaryPosEmb);
}  // namespace kernel
}  // namespace mindspore
