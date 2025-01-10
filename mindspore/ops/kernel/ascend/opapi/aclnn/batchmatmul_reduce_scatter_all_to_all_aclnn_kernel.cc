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
#include <cstdint>
#include <string>
#include "include/common/utils/utils.h"
#include "mindspore/ops/ops_utils/op_constants.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/opapi/aclnn/batchmatmul_reduce_scatter_all_to_all_aclnn_kernel.h"
#include "mindspore/ccsrc/transform/acl_ir/op_api_util.h"
namespace mindspore {
namespace kernel {
void BatchMatMulReduceScatterAlltoAllAscend::InitializeCommunicationAttributes() {
  group_ep_ = GetRequiredAttr<std::string>(kAttrGroupEp);
  group_tp_ = GetRequiredAttr<std::string>(kAttrGroupTp);
  hccl_inner_comm_ep_name_ = mindspore::transform::OpApiUtil::GetCommName(group_ep_);
  hccl_inner_comm_tp_name_ = mindspore::transform::OpApiUtil::GetCommName(group_tp_);
  ep_world_size_ = GetRequiredAttr<int64_t>(kAttrEpWorldSize);
  tp_world_size_ = GetRequiredAttr<int64_t>(kAttrTpWorldSize);
  y_shard_type_ = GetRequiredAttr<int64_t>(kAttrYShardType);
  transpose_weight_ = GetRequiredAttr<bool>(kAttrTransposeWeight);
}

void BatchMatMulReduceScatterAlltoAllAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                              const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);

  InitializeCommunicationAttributes();

  // support weight transposition
  input_x_ = std::pair<KernelTensor *, bool>(inputs[kIndex0], false);
  input_weight_ = std::pair<KernelTensor *, bool>(inputs[kIndex1], transpose_weight_);

  if (inputs.size() == kIndex2) {
    GetWorkspaceForResize(input_x_, input_weight_, nullptr, hccl_inner_comm_ep_name_, hccl_inner_comm_tp_name_,
                          ep_world_size_, tp_world_size_, y_shard_type_, outputs[kIndex0]);
  } else if (inputs.size() == kIndex3) {
    GetWorkspaceForResize(input_x_, input_weight_, inputs[kIndex2], hccl_inner_comm_ep_name_, hccl_inner_comm_tp_name_,
                          ep_world_size_, tp_world_size_, y_shard_type_, outputs[kIndex0]);
  } else {
    MS_LOG(ERROR) << "BatchMatMulReduceScatterAlltoAll: The size of inputs must be 2 or 3, but got " << inputs.size();
  }
}

bool BatchMatMulReduceScatterAlltoAllAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &workspace,
                                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  input_x_.first = inputs[kIndex0];
  input_weight_.first = inputs[kIndex1];

  if (inputs.size() == kIndex2) {
    RunOp(stream_ptr, workspace, input_x_, input_weight_, nullptr, hccl_inner_comm_ep_name_, hccl_inner_comm_tp_name_,
          ep_world_size_, tp_world_size_, y_shard_type_, outputs[kIndex0]);
  } else if (inputs.size() == kIndex3) {
    RunOp(stream_ptr, workspace, input_x_, input_weight_, inputs[kIndex2], hccl_inner_comm_ep_name_,
          hccl_inner_comm_tp_name_, ep_world_size_, tp_world_size_, y_shard_type_, outputs[kIndex0]);
  } else {
    MS_LOG(ERROR) << "BatchMatMulReduceScatter: The size of inputs must be 2 or 3, but got " << inputs.size();
    return false;
  }

  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(BatchMatMulReduceScatterAlltoAll, BatchMatMulReduceScatterAlltoAllAscend);
}  // namespace kernel
}  // namespace mindspore
