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

#include "kernel/ascend/pyboost/customize/moe_distribute_dispatch.h"
#include <cstdint>
#include <memory>
#include <vector>
#include <tuple>
#include <string>
#include "include/common/utils/utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "plugin/res_manager/ascend/collective/ascend_collective_comm_lib.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void MoeDistributeDispatchAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &x, const BaseTensorPtr &expert_ids,
  const Int64ImmPtr &ep_world_size, const Int64ImmPtr &ep_rank_id, const Int64ImmPtr &moe_expert_num,
  const std::optional<BaseTensorPtr> &expert_scales, const std::optional<BaseTensorPtr> &scales,
  const std::optional<BaseTensorPtr> &x_activate_mask, const std::optional<StringImmPtr> &group_ep,
  const std::optional<StringImmPtr> &group_tp, const Int64ImmPtr &tp_world_size, const Int64ImmPtr &tp_rank_id,
  const Int64ImmPtr &expert_shard_type, const Int64ImmPtr &shared_expert_num, const Int64ImmPtr &shared_expert_rank_num,
  const Int64ImmPtr &quant_mode, const Int64ImmPtr &global_bs, const Int64ImmPtr &expert_token_nums_type) {
  OpRunner::InferOpOutput(op, x, expert_ids, ep_world_size, ep_rank_id, moe_expert_num, expert_scales, scales,
                          x_activate_mask, group_ep, group_tp, tp_world_size, tp_rank_id, expert_shard_type,
                          shared_expert_num, shared_expert_rank_num, quant_mode, global_bs, expert_token_nums_type);
  // Convert ValuePtr to c++ scalar
  int64_t ep_world_size_imm = GetValue<int64_t>(ep_world_size);
  int64_t ep_rank_id_imm = GetValue<int64_t>(ep_rank_id);
  int64_t moe_expert_num_imm = GetValue<int64_t>(moe_expert_num);
  std::string group_ep_str = group_ep.has_value() ? GetValue<std::string>(group_ep.value()) : kHcclWorldGroup;
  std::string group_ep_imm = device::ascend::OpApiUtil::GetCommName(group_ep_str);
  std::string group_tp_str = group_tp.has_value() ? GetValue<std::string>(group_tp.value()) : "";
  std::string group_tp_imm = group_tp_str.empty() ? "" : device::ascend::OpApiUtil::GetCommName(group_ep_str);
  int64_t tp_world_size_imm = GetValue<int64_t>(tp_world_size);
  int64_t tp_rank_id_imm = GetValue<int64_t>(tp_rank_id);
  int64_t expert_shard_type_imm = GetValue<int64_t>(expert_shard_type);
  int64_t shared_expert_num_imm = GetValue<int64_t>(shared_expert_num);
  int64_t shared_expert_rank_num_imm = GetValue<int64_t>(shared_expert_rank_num);
  int64_t quant_mode_imm = GetValue<int64_t>(quant_mode);
  int64_t global_bs_imm = GetValue<int64_t>(global_bs);
  int64_t expert_token_nums_type_imm = GetValue<int64_t>(expert_token_nums_type);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x, expert_ids, scales, x_activate_mask,
                                expert_scales);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x, expert_ids, scales, x_activate_mask, expert_scales, group_ep_imm, ep_world_size_imm, ep_rank_id_imm,
     moe_expert_num_imm, group_tp_imm, tp_world_size_imm, tp_rank_id_imm, expert_shard_type_imm, shared_expert_num_imm,
     shared_expert_rank_num_imm, quant_mode_imm, global_bs_imm, expert_token_nums_type_imm]() {
      MS_LOG(DEBUG) << "Run device task MoeDistributeDispatch start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(op->device_context(), x, expert_ids, scales, x_activate_mask, expert_scales);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(op->device_context(), outputs);
      LAUNCH_ACLNN(aclnnMoeDistributeDispatch, device_context, op->stream_id(), x, expert_ids, scales, x_activate_mask,
                   expert_scales, group_ep_imm, ep_world_size_imm, ep_rank_id_imm, moe_expert_num_imm, group_tp_imm,
                   tp_world_size_imm, tp_rank_id_imm, expert_shard_type_imm, shared_expert_num_imm,
                   shared_expert_rank_num_imm, quant_mode_imm, global_bs_imm, expert_token_nums_type_imm, outputs[0],
                   outputs[1], outputs[2], outputs[3], outputs[4], outputs[5], outputs[6]);
      MS_LOG(DEBUG) << "Run device task MoeDistributeDispatch end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
