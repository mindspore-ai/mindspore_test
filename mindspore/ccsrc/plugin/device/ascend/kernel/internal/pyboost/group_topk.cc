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

#include "plugin/device/ascend/kernel/internal/pyboost/group_topk.h"

#include "common/kernel.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr GroupTopk::CreateKernel(const internal::InputsImmutableInfoList &inputs,
                                                const internal::OutputsImmutableInfoList &outputs) {
  return internal::CreateGroupTopkOp(inputs, outputs, param_, internal::kInternalGroupTopkOpName);
}

void GroupTopk::Call(const std::shared_ptr<pyboost::OpRunner> &op, const uint64_t &op_key, const uint64_t &tiling_key,
                     const BaseTensorPtr &token, const BaseTensorPtr &idx_arr, const int64_t &group_num,
                     const int64_t &k, const int64_t &k_inner) {
  std::vector<BaseTensorPtr> inputs = {token, idx_arr};
  std::vector<BaseTensorPtr> outputs = {};
  TransInternalShapes(inputs, outputs);

  param_.group_num = static_cast<int32_t>(group_num);
  param_.k = static_cast<int32_t>(k);
  param_.k_inner = static_cast<int32_t>(k_inner);

  GetOrCreateKernel(op, op_key, tiling_key, inputs, outputs);

  LAUNCH_INTERNAL(kernel_name_, op, internal_op_, inputs, outputs, tiling_info_);
}
MS_INTERNAL_KERNEL_INFO_FACTORY_REG(GroupTopk, GroupTopk);
}  // namespace kernel
}  // namespace mindspore
