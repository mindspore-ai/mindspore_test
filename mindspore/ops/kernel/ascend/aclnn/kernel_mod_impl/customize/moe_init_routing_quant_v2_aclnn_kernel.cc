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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/moe_init_routing_quant_v2_aclnn_kernel.h"

#include <algorithm>
#include <vector>
#include <string>

#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace moe_init_routing_quant_v2 {
void MoeInitRoutingQuantV2Ascend::InitInputAttributes(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  active_num_ = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  expert_capacity_ = inputs[kIndex3]->GetValueWithCheck<int64_t>();
  expert_num_ = inputs[kIndex4]->GetValueWithCheck<int64_t>();
  drop_pad_mode_ = inputs[kIndex5]->GetValueWithCheck<int64_t>();
  expert_tokens_count_or_cumsum_flag_ = inputs[kIndex6]->GetValueWithCheck<int64_t>();
  expert_tokens_before_capacity_flag_ = inputs[kIndex7]->GetValueWithCheck<bool>();
  quant_mode_ = inputs[kIndex8]->GetValueWithCheck<int64_t>();
}

void MoeInitRoutingQuantV2Ascend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);
  InitInputAttributes(inputs, outputs);
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex9], inputs[kIndex10], active_num_,
                        expert_capacity_, expert_num_, drop_pad_mode_, expert_tokens_count_or_cumsum_flag_,
                        expert_tokens_before_capacity_flag_, quant_mode_, outputs[kIndex0], outputs[kIndex1],
                        outputs[kIndex2], outputs[kIndex3], outputs[kIndex4]);
}

bool MoeInitRoutingQuantV2Ascend::Launch(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex9], inputs[kIndex10], active_num_,
        expert_capacity_, expert_num_, drop_pad_mode_, expert_tokens_count_or_cumsum_flag_,
        expert_tokens_before_capacity_flag_, quant_mode_, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2],
        outputs[kIndex3], outputs[kIndex4]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MoeInitRoutingQuantV2, MoeInitRoutingQuantV2Ascend);
}  // namespace moe_init_routing_quant_v2
}  // namespace kernel
}  // namespace mindspore
