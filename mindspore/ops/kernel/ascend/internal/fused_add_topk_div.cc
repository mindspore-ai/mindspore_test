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

#include "kernel/ascend/internal/fused_add_topk_div.h"

#include <memory>
#include "include/runtime/hardware_abstract/kernel_base/kernel.h"
#include "kernel/ascend/internal/internal_kernel_in_out_map.h"

namespace mindspore {
namespace kernel {
internal::InternalOpPtr InternalFusedAddTopKDiv::CreateKernel(const internal::InputsImmutableInfoList &inputs_ii,
                                                              const internal::OutputsImmutableInfoList &outputs_ii,
                                                              const std::vector<KernelTensor *> &ms_inputs,
                                                              const std::vector<KernelTensor *> &ms_outputs) {
  internal::FusedAddTopkDivParam param;
  auto group_num = ms_inputs.at(kIndex2);
  auto group_topk = ms_inputs.at(kIndex3);
  auto n = ms_inputs.at(kIndex4);
  auto k = ms_inputs.at(kIndex5);
  auto activate_type = ms_inputs.at(kIndex6);
  auto is_norm = ms_inputs.at(kIndex7);
  auto scale = ms_inputs.at(kIndex8);
  auto enableExpertMapping = ms_inputs.at(kIndex11);
  if (group_num->dtype_id() == TypeId::kNumberTypeInt64 && group_topk->dtype_id() == TypeId::kNumberTypeInt64 &&
      n->dtype_id() == TypeId::kNumberTypeInt64 && k->dtype_id() == TypeId::kNumberTypeInt64 &&
      activate_type->dtype_id() == TypeId::kNumberTypeInt64 && is_norm->dtype_id() == TypeId::kNumberTypeBool &&
      scale->dtype_id() == TypeId::kNumberTypeFloat32) {
    param.group_num = static_cast<int32_t>(group_num->GetValue<int64_t>().value());
    param.group_topk = static_cast<int32_t>(group_topk->GetValue<int64_t>().value());
    param.n = static_cast<int32_t>(n->GetValue<int64_t>().value());
    param.k = static_cast<int32_t>(k->GetValue<int64_t>().value());
    param.activate_type = static_cast<int32_t>(activate_type->GetValue<int64_t>().value());
    param.is_norm = is_norm->GetValue<bool>().value();
    param.scale = scale->GetValue<float>().value();
    param.enableExpertMapping = enableExpertMapping->GetValue<bool>().value();
  } else {
    MS_LOG(EXCEPTION) << "FusedAddTopKDiv [group_num, group_topk, n, k, activate_type, is_norm, scale]'s dtype wrong";
  }
  return internal::CreateFusedAddTopkDivOp(inputs_ii, outputs_ii, param, internal::kInternalFusedAddTopkDivOpName);
}

MS_INTERNAL_KERNEL_FACTORY_REG(FusedAddTopKDiv, internal::kInternalFusedAddTopkDivOpName, InternalFusedAddTopKDiv);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FusedAddTopKDiv, INPUT_NUM_2, INDEX_0, INDEX_1);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FusedAddTopKDiv, INPUT_NUM_4, INDEX_0, INDEX_1, INDEX_9, INDEX_10);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FusedAddTopKDiv, OUTPUT_NUM_2, INDEX_0, INDEX_1);
}  // namespace kernel
}  // namespace mindspore
