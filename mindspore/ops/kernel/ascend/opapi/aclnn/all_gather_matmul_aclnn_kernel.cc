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
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "plugin/device/ascend/hal/hardware/ascend_collective_comm/ascend_collective_comm_lib.h"
#include "kernel/ascend/opapi/aclnn/all_gather_matmul_aclnn_kernel.h"
#include "plugin/device/ascend/acl_ir/op_api_util.h"
#include "mindspore/ops/infer/ops_func_impl/all_gather_matmul.h"

namespace mindspore {
namespace kernel {
void AllGatherMatmulAscend::InitializeInputs(const std::vector<KernelTensor *> &inputs) {
  trans_input_ = inputs[mindspore::ops::kAllGatherMatmulInputTransInputIndex]->GetValueWithCheck<bool>();
  trans_x2_ = inputs[mindspore::ops::kAllGatherMatmulInputTransX2Index]->GetValueWithCheck<bool>();
  input_ = std::pair<KernelTensor *, bool>(inputs[mindspore::ops::kAllGatherMatmulInputInputIndex], trans_input_);
  x2_ = std::pair<KernelTensor *, bool>(inputs[mindspore::ops::kAllGatherMatmulInputX2Index], trans_x2_);
  group_ = inputs[mindspore::ops::kAllGatherMatmulInputGroupIndex]->GetValueWithCheck<std::string>();
  hccl_inner_comm_name_ = mindspore::device::ascend::OpApiUtil::GetCommName(group_);
  gather_index_ = inputs[mindspore::ops::kAllGatherMatmulInputGatherIndexIndex]->GetValueWithCheck<int64_t>();
  comm_turn_ = inputs[mindspore::ops::kAllGatherMatmulInputCommTurnIndex]->GetValueWithCheck<int64_t>();
}

void AllGatherMatmulAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);
  InitializeInputs(inputs);
  GetWorkspaceForResize(input_, x2_, nullptr, hccl_inner_comm_name_, gather_index_, comm_turn_, stream_mode_,
                        outputs[mindspore::ops::kAllGatherMatmulOutputYIndex],
                        outputs[mindspore::ops::kAllGatherMatmulOutputGatherOutIndex]);
}

bool AllGatherMatmulAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  // The following two lines are nessisary; deleting them will cause an error: "Sync default stream failed."
  input_.first = inputs[mindspore::ops::kAllGatherMatmulInputInputIndex];
  x2_.first = inputs[mindspore::ops::kAllGatherMatmulInputX2Index];
  RunOp(stream_ptr, workspace, input_, x2_, nullptr, hccl_inner_comm_name_, gather_index_, comm_turn_, stream_mode_,
        outputs[mindspore::ops::kAllGatherMatmulOutputYIndex],
        outputs[mindspore::ops::kAllGatherMatmulOutputGatherOutIndex]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AllGatherMatmul, AllGatherMatmulAscend);
}  // namespace kernel
}  // namespace mindspore
