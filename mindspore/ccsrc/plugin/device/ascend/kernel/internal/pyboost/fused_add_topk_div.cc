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

#include "plugin/device/ascend/kernel/internal/pyboost/fused_add_topk_div.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr FusedAddTopKDiv::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                      const internal::OutputsImmutableInfoList &outputs) {
  return internal::CreateFusedAddTopkDivOp(inputs, outputs, param_, internal::kInternalFusedAddTopkDivOpName);
}

void FusedAddTopKDiv::Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key,
                           const uint64_t &tiling_key, const TensorPtr &x, const TensorPtr &add_num,
                           const int64_t &group_num, const int64_t &group_topk, const int64_t &n, const int64_t &k,
                           const int64_t &activate_type, const bool &is_norm, const float &scale,
                           const std::optional<TensorPtr> &mapping_num, const std::optional<TensorPtr> &mapping_table,
                           const bool &enable_expert_mapping) {
  TensorPtrList inputs = {x, add_num, mapping_num.has_value() ? mapping_num.value() : nullptr,
                          mapping_table.has_value() ? mapping_table.value() : nullptr};

  TensorPtrList outputs = op->outputs();
  TransInternalShapes(inputs, outputs);

  param_.group_num = static_cast<int32_t>(group_num);
  param_.group_topk = static_cast<int32_t>(group_topk);
  param_.n = static_cast<int32_t>(n);
  param_.k = static_cast<int32_t>(k);
  param_.activate_type = static_cast<int32_t>(activate_type);
  param_.is_norm = static_cast<bool>(is_norm);
  param_.scale = scale;
  param_.enableExpertMapping = static_cast<bool>(enable_expert_mapping);

  GetOrCreateKernel(op, op_key, tiling_key, inputs, outputs);
  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(FusedAddTopKDiv, FusedAddTopKDiv);
}  // namespace kernel
}  // namespace mindspore
