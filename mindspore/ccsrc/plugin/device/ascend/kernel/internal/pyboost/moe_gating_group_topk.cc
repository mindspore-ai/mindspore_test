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

#include "plugin/device/ascend/kernel/internal/pyboost/moe_gating_group_topk.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr MoeGatingGroupTopK::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                         const internal::OutputsImmutableInfoList &outputs) {
  return internal::CreateMoeGatingGroupTopKOp(inputs, outputs, param_, internal::kInternalMoeGatingGroupTopKOpName);
}

void MoeGatingGroupTopK::Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key,
                              const uint64_t &tiling_key, const BaseTensorPtr &x_tensor,
                              const std::optional<BaseTensorPtr> &bias_tensor, const int64_t &k, const int64_t &k_group,
                              const int64_t &group_count, const int64_t &group_select_mode, const int64_t &renorm,
                              const int64_t &norm_type, const bool &out_flag, const float &routed_scaling_factor,
                              const float &eps) {
  BaseTensorPtrList inputs = {x_tensor, bias_tensor.has_value() ? bias_tensor.value() : nullptr};
  BaseTensorPtrList outputs = op->outputs();
  TransInternalShapes(inputs, outputs);
  param_.k = static_cast<int32_t>(k);
  param_.k_group = static_cast<int32_t>(k_group);
  param_.group_count = static_cast<int32_t>(group_count);
  param_.group_select_mode = static_cast<int32_t>(group_select_mode);
  param_.renorm = static_cast<int32_t>(renorm);
  param_.norm_type = static_cast<int32_t>(norm_type);
  param_.out_flag = out_flag;
  param_.routed_scaling_factor = routed_scaling_factor;
  param_.eps = eps;
  GetOrCreateKernel(op, op_key, tiling_key, inputs, outputs);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(MoeGatingGroupTopK, MoeGatingGroupTopK);
}  // namespace kernel
}  // namespace mindspore
