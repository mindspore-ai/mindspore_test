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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/moe_distribute_dispatch_aclnn_kernel.h"

#include <algorithm>
#include <vector>
#include <string>

#include "ir/tensor.h"
#include "plugin/ascend/res_manager/collective/ascend_collective_comm_lib.h"

namespace mindspore {
namespace kernel {
namespace moe_distribute_dispatch {
namespace {
constexpr size_t kInputSize = 18;

constexpr size_t kX = 0;
constexpr size_t kExpertIds = 1;
constexpr size_t kScales = 6;
constexpr size_t kXActiveMask = 7;
constexpr size_t kExpertScales = 5;
constexpr size_t kGroupEp = 8;
constexpr size_t kEpWorldSize = 2;
constexpr size_t kEpRankId = 3;
constexpr size_t kMoeExpertNum = 4;
constexpr size_t kGroupTp = 9;
constexpr size_t kTpWorldSize = 10;
constexpr size_t kTpRankId = 11;
constexpr size_t kExpertShardType = 12;
constexpr size_t kSharedExpertNum = 13;
constexpr size_t kSharedExpertRankNum = 14;
constexpr size_t kQuantMode = 15;
constexpr size_t kGlobalBs = 16;
constexpr size_t kExpertTokenNumsType = 17;

constexpr size_t kExpandX = 0;
constexpr size_t kDynamicScales = 1;
constexpr size_t kExpandIdx = 2;
constexpr size_t kExpertTokenNums = 3;
constexpr size_t kEpRecvCounts = 4;
constexpr size_t kTpRecvCounts = 5;
constexpr size_t kExpandScales = 6;
}  // namespace

void MoeDistributeDispatchAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  if (inputs.size() != kInputSize) {
    MS_LOG(EXCEPTION) << "For MoeDistributeDispatch, the input size should be equal to " << kInputSize << ", but got "
                      << inputs.size();
  }
  ep_world_size_ = inputs[kEpWorldSize]->GetValueWithCheck<int64_t>();
  ep_rank_id_ = inputs[kEpRankId]->GetValueWithCheck<int64_t>();
  moe_expert_num_ = inputs[kMoeExpertNum]->GetValueWithCheck<int64_t>();

  auto group_ep = inputs[kGroupEp]->GetOptionalValueWithCheck<std::string>();
  group_ep_ = group_ep.has_value() ? OpApiUtil::GetCommName(group_ep.value()) : OpApiUtil::GetCommName(kHcclWorldGroup);
  auto group_tp = inputs[kGroupTp]->GetOptionalValueWithCheck<std::string>();
  group_tp_ = group_tp.has_value() ? OpApiUtil::GetCommName(group_tp.value()) : "";
  auto tp_world_size = inputs[kTpWorldSize]->GetOptionalValueWithCheck<int64_t>();
  tp_world_size_ = tp_world_size.has_value() ? tp_world_size.value() : 0;
  auto tp_rank_id = inputs[kTpRankId]->GetOptionalValueWithCheck<int64_t>();
  tp_rank_id_ = tp_rank_id.has_value() ? tp_rank_id.value() : 0;
  auto expert_shard_type = inputs[kExpertShardType]->GetOptionalValueWithCheck<int64_t>();
  expert_shard_type_ = expert_shard_type.has_value() ? expert_shard_type.value() : 0;
  auto shared_expert_num = inputs[kSharedExpertNum]->GetOptionalValueWithCheck<int64_t>();
  shared_expert_num_ = shared_expert_num.has_value() ? shared_expert_num.value() : 0;
  auto shared_expert_rank_num = inputs[kSharedExpertRankNum]->GetOptionalValueWithCheck<int64_t>();
  shared_expert_rank_num_ = shared_expert_rank_num.has_value() ? shared_expert_rank_num.value() : 0;
  auto quant_mode = inputs[kQuantMode]->GetOptionalValueWithCheck<int64_t>();
  quant_mode_ = quant_mode.has_value() ? quant_mode.value() : 0;
  auto global_bs = inputs[kGlobalBs]->GetOptionalValueWithCheck<int64_t>();
  global_bs_ = global_bs.has_value() ? global_bs.value() : 0;
  auto expert_token_nums_type = inputs[kExpertTokenNumsType]->GetOptionalValueWithCheck<int64_t>();
  expert_token_nums_type_ = expert_token_nums_type.has_value() ? expert_token_nums_type.value() : 0;

  GetWorkspaceForResize(inputs[kX], inputs[kExpertIds], inputs[kScales], inputs[kXActiveMask], inputs[kExpertScales],
                        group_ep_, ep_world_size_, ep_rank_id_, moe_expert_num_, group_tp_, tp_world_size_, tp_rank_id_,
                        expert_shard_type_, shared_expert_num_, shared_expert_rank_num_, quant_mode_, global_bs_,
                        expert_token_nums_type_, outputs[kExpandX], outputs[kDynamicScales], outputs[kExpandIdx],
                        outputs[kExpertTokenNums], outputs[kEpRecvCounts], outputs[kTpRecvCounts],
                        outputs[kExpandScales]);
}

bool MoeDistributeDispatchAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kX], inputs[kExpertIds], inputs[kScales], inputs[kXActiveMask],
        inputs[kExpertScales], group_ep_, ep_world_size_, ep_rank_id_, moe_expert_num_, group_tp_, tp_world_size_,
        tp_rank_id_, expert_shard_type_, shared_expert_num_, shared_expert_rank_num_, quant_mode_, global_bs_,
        expert_token_nums_type_, outputs[kExpandX], outputs[kDynamicScales], outputs[kExpandIdx],
        outputs[kExpertTokenNums], outputs[kEpRecvCounts], outputs[kTpRecvCounts], outputs[kExpandScales]);

  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MoeDistributeDispatch, MoeDistributeDispatchAscend);
}  // namespace moe_distribute_dispatch
}  // namespace kernel
}  // namespace mindspore
