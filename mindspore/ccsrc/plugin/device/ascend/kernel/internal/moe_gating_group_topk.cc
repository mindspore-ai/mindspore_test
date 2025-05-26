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

#include "plugin/device/ascend/kernel/internal/moe_gating_group_topk.h"
#include <vector>
#include <memory>
#include "common/kernel.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalMoeGatingGroupTopK::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                                 const internal::OutputsImmutableInfoList &outputs_ii,
                                                                 const std::vector<KernelTensor *> &ms_inputs,
                                                                 const std::vector<KernelTensor *> &ms_outputs) {
  internal::MoeGatingGroupTopKParam param;
  auto k = ms_inputs.at(kIndex2);
  auto k_group = ms_inputs.at(kIndex3);
  auto group_count = ms_inputs.at(kIndex4);
  auto group_select_mode = ms_inputs.at(kIndex5);
  auto renorm = ms_inputs.at(kIndex6);
  auto norm_type = ms_inputs.at(kIndex7);
  auto out_flag = ms_inputs.at(kIndex8);
  auto routed_scaling_factor = ms_inputs.at(kIndex9);
  auto eps = ms_inputs.at(kIndex10);
  if (k->dtype_id() == TypeId::kNumberTypeInt64 && k_group->dtype_id() == TypeId::kNumberTypeInt64 &&
      group_count->dtype_id() == TypeId::kNumberTypeInt64 &&
      group_select_mode->dtype_id() == TypeId::kNumberTypeInt64 && renorm->dtype_id() == TypeId::kNumberTypeInt64 &&
      norm_type->dtype_id() == TypeId::kNumberTypeInt64 && out_flag->dtype_id() == TypeId::kNumberTypeBool &&
      routed_scaling_factor->dtype_id() == TypeId::kNumberTypeFloat32 &&
      eps->dtype_id() == TypeId::kNumberTypeFloat32) {
    param.k = static_cast<int32_t>(k->GetValue<int64_t>().value());
    param.k_group = static_cast<int32_t>(k_group->GetValue<int64_t>().value());
    param.group_count = static_cast<int32_t>(group_count->GetValue<int64_t>().value());
    param.group_select_mode = static_cast<int32_t>(group_select_mode->GetValue<int64_t>().value());
    param.renorm = static_cast<int32_t>(renorm->GetValue<int64_t>().value());
    param.norm_type = static_cast<int32_t>(norm_type->GetValue<int64_t>().value());
    param.out_flag = out_flag->GetValue<bool>().value();
    param.routed_scaling_factor = routed_scaling_factor->GetValue<float>().value();
    param.eps = eps->GetValue<float>().value();
  } else {
    MS_LOG(EXCEPTION) << "MoeGatingGroupTopK inputs[k, k_group, group_count, group_select_mode, renorm, norm_type, "
                         "out_flag, routed_scaling_factor, eps]'s dtype should be [kNumberTypeInt64, kNumberTypeInt64, "
                         "kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeInt64, kNumberTypeBool, "
                         "kNumberTypeFloat32, kNumberTypeFloat32], but got ["
                      << TypeIdToString(k->dtype_id()) << ", " << TypeIdToString(k_group->dtype_id()) << ", "
                      << TypeIdToString(group_count->dtype_id()) << ", "
                      << TypeIdToString(group_select_mode->dtype_id()) << ", " << TypeIdToString(renorm->dtype_id())
                      << ", " << TypeIdToString(norm_type->dtype_id()) << ", " << TypeIdToString(out_flag->dtype_id())
                      << ", " << TypeIdToString(routed_scaling_factor->dtype_id()) << ", "
                      << TypeIdToString(eps->dtype_id()) << "]";
  }
  return internal::CreateMoeGatingGroupTopKOp(inputs_ii, outputs_ii, param,
                                              internal::kInternalMoeGatingGroupTopKOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(MoeGatingGroupTopK, internal::kInternalMoeGatingGroupTopKOpName,
                               InternalMoeGatingGroupTopK);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(MoeGatingGroupTopK, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(MoeGatingGroupTopK, OUTPUT_NUM_3, INDEX_0, INDEX_1, INDEX_2);
}  // namespace kernel
}  // namespace mindspore
