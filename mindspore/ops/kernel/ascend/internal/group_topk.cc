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

#include "kernel/ascend/internal/group_topk.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalGroupTopk::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                        const internal::OutputsImmutableInfoList &outputs_ii,
                                                        const std::vector<KernelTensor *> &ms_inputs,
                                                        const std::vector<KernelTensor *> &ms_outputs) {
  internal::GroupTopkParam param;
  auto input_group_num = ms_inputs.at(kIndex2);
  auto input_k = ms_inputs.at(kIndex3);
  auto input_k_inner = ms_inputs.at(kIndex4);
  if (input_group_num->dtype_id() == TypeId::kNumberTypeInt64 && input_k->dtype_id() == TypeId::kNumberTypeInt64 &&
      input_k_inner->dtype_id() == TypeId::kNumberTypeInt64) {
    param.group_num = static_cast<int32_t>(input_group_num->GetValue<int64_t>().value());
    param.k = static_cast<int32_t>(input_k->GetValue<int64_t>().value());
    param.k_inner = static_cast<int32_t>(input_k_inner->GetValue<int64_t>().value());
  } else {
    MS_LOG(EXCEPTION) << "GroupTopk [group_num, k, k_inner]'s dtype should all be kNumberTypeInt64, but is ["
                      << TypeIdToString(input_group_num->dtype_id()) << ", " << TypeIdToString(input_k->dtype_id())
                      << "," << TypeIdToString(input_k_inner->dtype_id()) << "]";
  }
  return internal::CreateGroupTopkOp(inputs_ii, outputs_ii, param, internal::kInternalGroupTopkOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(GroupTopk, internal::kInternalGroupTopkOpName, InternalGroupTopk);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(GroupTopk, INPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
