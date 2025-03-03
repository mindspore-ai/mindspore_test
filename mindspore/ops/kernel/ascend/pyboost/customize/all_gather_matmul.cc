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

#include <memory>
#include <string>
#include "kernel/ascend/pyboost/customize/all_gather_matmul.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "kernel/ascend/pyboost/auto_generate/transpose.h"
#include "plugin/device/ascend/acl_ir/op_api_util.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
ValueTuplePtr GetTransposePerm(const BaseTensorPtr &tensor) {
  std::vector<ValuePtr> perm(tensor->shape().size());
  perm[kDim0] = MakeValue(static_cast<int64_t>(kDim1));
  perm[kDim1] = MakeValue(static_cast<int64_t>(kDim0));
  return std::make_shared<ValueTuple>(perm);
}
}  // namespace

std::vector<tensor::BaseTensorPtr> AllGatherMatmulAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input, const BaseTensorPtr &x2, const StringImmPtr &group,
  const Int64ImmPtr &world_size, const std::optional<BaseTensorPtr> &bias, const Int64ImmPtr &gather_index,
  const BoolImmPtr &gather_output, const Int64ImmPtr &comm_turn, const BoolImmPtr &trans_input,
  const BoolImmPtr &trans_x2) {
  MS_LOG(DEBUG) << op->primitive()->name() << " call start";

  OpRunner::InferOpOutput(op, input, x2, group, world_size, bias, gather_index, gather_output, comm_turn, trans_input,
                          trans_x2);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input, x2, bias);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  auto group_imm = GetValue<std::string>(group);
  auto world_size_imm = GetValue<int64_t>(world_size);
  auto gather_index_imm = GetValue<int64_t>(gather_index);
  auto comm_turn_imm = GetValue<int64_t>(comm_turn);
  auto trans_input_imm = GetValue<bool>(trans_input);
  auto trans_x2_imm = GetValue<bool>(trans_x2);

  auto hccl_inner_comm_name_imm = mindspore::device::ascend::OpApiUtil::GetCommName(group_imm);
  mindspore::device::ascend::OpApiUtil::CheckWorldSize(group_imm, world_size_imm, op->primitive()->name());
  BaseTensorPtr input_ = input;
  BaseTensorPtr x2_ = x2;
  const auto &device_name = op->device_context()->device_context_key_.device_name_;
  auto transpose_op = CREATE_PYBOOST_OP(Transpose, device_name);
  if (trans_input_imm) {
    input_ = transpose_op->Call(input, GetTransposePerm(input));
  }
  if (trans_x2_imm) {
    x2_ = transpose_op->Call(x2, GetTransposePerm(x2));
  }

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, input_, x2_, hccl_inner_comm_name_imm, bias, gather_index_imm, comm_turn_imm]() {
      MS_LOG(DEBUG) << op->primitive()->name() << " run device task start";
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      PyBoostUtils::MallocOpInputs(device_context, input_, x2_, bias);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      constexpr int64_t stream_mode = 1;
      if (bias.has_value()) {
        LAUNCH_ACLNN(aclnnAllGatherMatmul, device_context, op->stream_id(), input_, x2_, bias, hccl_inner_comm_name_imm,
                     gather_index_imm, comm_turn_imm, stream_mode, outputs[0], outputs[1]);
      } else {
        LAUNCH_ACLNN(aclnnAllGatherMatmul, device_context, op->stream_id(), input_, x2_, nullptr,
                     hccl_inner_comm_name_imm, gather_index_imm, comm_turn_imm, stream_mode, outputs[0], outputs[1]);
      }
      MS_LOG(DEBUG) << op->primitive()->name() << " run device task end";
    }));
  return op->outputs();
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
