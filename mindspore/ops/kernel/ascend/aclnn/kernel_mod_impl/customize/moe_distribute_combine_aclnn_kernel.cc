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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/moe_distribute_combine_aclnn_kernel.h"

#include <algorithm>
#include <vector>
#include <string>

#include "ir/tensor.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"

namespace mindspore {
namespace kernel {
namespace moe_distribute_combine {
namespace {
constexpr size_t kInputSize = 25;
constexpr size_t kExpandXIndex = 0;
constexpr size_t kExpertIdsIndex = 1;
constexpr size_t kExpandIdxIndex = 2;
constexpr size_t kEpSendIndex = 3;
constexpr size_t kExpertScaleIndex = 4;
constexpr size_t kTpSendIndex = 8;
constexpr size_t kActivateMaskIndex = 9;
constexpr size_t kActivateScaleIndex = 10;
constexpr size_t kWeightScaleIndex = 11;
constexpr size_t kGroupListIndex = 12;
constexpr size_t kExpandScaleIndex = 13;
}  // namespace

void MoeDistributeCombineAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kInputSize) {
    MS_LOG(EXCEPTION) << "For MoeDistributeCombine, the input size should be equal to " << kInputSize << ", but got "
                      << inputs.size();
  }

  ep_world_size_ = inputs[kIndex5]->GetValueWithCheck<int64_t>();
  ep_rank_id_ = inputs[kIndex6]->GetValueWithCheck<int64_t>();
  moe_expert_num_ = inputs[kIndex7]->GetValueWithCheck<int64_t>();

  auto group_ep = inputs[kIndex14]->GetOptionalValueWithCheck<std::string>();
  group_ep_ = group_ep.has_value() ? OpApiUtil::GetCommName(group_ep.value()) : OpApiUtil::GetCommName(kHcclWorldGroup);
  auto group_tp = inputs[kIndex15]->GetOptionalValueWithCheck<std::string>();
  group_tp_ = group_tp.has_value() ? OpApiUtil::GetCommName(group_tp.value()) : "";
  auto tp_world_size = inputs[kIndex16]->GetOptionalValueWithCheck<int64_t>();
  tp_world_size_ = tp_world_size.has_value() ? tp_world_size.value() : 0;
  auto tp_rank_id = inputs[kIndex17]->GetOptionalValueWithCheck<int64_t>();
  tp_rank_id_ = tp_rank_id.has_value() ? tp_rank_id.value() : 0;
  auto expert_shard_type = inputs[kIndex18]->GetOptionalValueWithCheck<int64_t>();
  expert_shard_type_ = expert_shard_type.has_value() ? expert_shard_type.value() : 0;
  auto shard_expert_num = inputs[kIndex19]->GetOptionalValueWithCheck<int64_t>();
  shard_expert_num_ = shard_expert_num.has_value() ? shard_expert_num.value() : 0;
  auto shared_expert_rank_num = inputs[kIndex20]->GetOptionalValueWithCheck<int64_t>();
  shared_expert_rank_num_ = shared_expert_rank_num.has_value() ? shared_expert_rank_num.value() : 0;
  auto global_bs = inputs[kIndex21]->GetOptionalValueWithCheck<int64_t>();
  global_bs_ = global_bs.has_value() ? global_bs.value() : 0;
  auto out_dtype = inputs[kIndex22]->GetOptionalValueWithCheck<int64_t>();
  out_dtype_ = out_dtype.has_value() ? out_dtype.value() : 0;
  auto common_quant_mode = inputs[kIndex23]->GetOptionalValueWithCheck<int64_t>();
  common_quant_mode_ = common_quant_mode.has_value() ? common_quant_mode.value() : 0;
  auto group_list_type = inputs[kIndex24]->GetOptionalValueWithCheck<int64_t>();
  group_list_type_ = group_list_type.has_value() ? group_list_type.value() : 0;

  GetWorkspaceForResize(inputs[kExpandXIndex], inputs[kExpertIdsIndex], inputs[kExpandIdxIndex], inputs[kEpSendIndex],
                        inputs[kExpertScaleIndex], inputs[kTpSendIndex], inputs[kActivateMaskIndex],
                        inputs[kActivateScaleIndex], inputs[kWeightScaleIndex], inputs[kGroupListIndex],
                        inputs[kExpandScaleIndex], group_ep_, ep_world_size_, ep_rank_id_, moe_expert_num_, group_tp_,
                        tp_world_size_, tp_rank_id_, expert_shard_type_, shard_expert_num_, shared_expert_rank_num_,
                        global_bs_, out_dtype_, common_quant_mode_, group_list_type_, outputs[kIndex0]);
}

bool MoeDistributeCombineAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kExpandXIndex], inputs[kExpertIdsIndex], inputs[kExpandIdxIndex],
        inputs[kEpSendIndex], inputs[kExpertScaleIndex], inputs[kTpSendIndex], inputs[kActivateMaskIndex],
        inputs[kActivateScaleIndex], inputs[kWeightScaleIndex], inputs[kGroupListIndex], inputs[kExpandScaleIndex],
        group_ep_, ep_world_size_, ep_rank_id_, moe_expert_num_, group_tp_, tp_world_size_, tp_rank_id_,
        expert_shard_type_, shard_expert_num_, shared_expert_rank_num_, global_bs_, out_dtype_, common_quant_mode_,
        group_list_type_, outputs[kIndex0]);

  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MoeDistributeCombine, MoeDistributeCombineAscend);
}  // namespace moe_distribute_combine
}  // namespace kernel
}  // namespace mindspore
