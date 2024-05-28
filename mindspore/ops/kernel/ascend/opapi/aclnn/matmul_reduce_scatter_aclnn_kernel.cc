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
#include "kernel/ascend/opapi/aclnn/matmul_reduce_scatter_aclnn_kernel.h"
namespace mindspore {
namespace kernel {
void MatmulReduceScatterAscend::InitializeCommonAttributes() {
  trans_a_ = GetRequiredAttr<bool>(kAttrIsTransA);
  trans_b_ = GetRequiredAttr<bool>(kAttrIsTransB);
  group_ = GetRequiredAttr<std::string>(kAttrGroup);
  hccl_inner_comm_name_ = GetCommName(group_);
  reduce_op_ = GetRequiredAttr<std::string>(kAttrReduceOp);
  comm_turn_ = GetRequiredAttr<int64_t>(kAttrCommTurn);
}

void MatmulReduceScatterAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);

  InitializeCommonAttributes();

  // Support B matrix transposition
  input_a_ = std::pair<KernelTensor *, bool>(inputs[kIndex0], trans_a_);
  input_b_ = std::pair<KernelTensor *, bool>(inputs[kIndex1], trans_b_);
  GetWorkspaceForResize(input_a_, input_b_, nullptr, hccl_inner_comm_name_, reduce_op_, comm_turn_, stream_mode_,
                        outputs[kIndex0]);
}

bool MatmulReduceScatterAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  input_a_.first = inputs[kIndex0];
  input_b_.first = inputs[kIndex1];
  RunOp(stream_ptr, workspace, input_a_, input_b_, nullptr, hccl_inner_comm_name_, reduce_op_, comm_turn_, stream_mode_,
        outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MatmulReduceScatter, MatmulReduceScatterAscend);
}  // namespace kernel
}  // namespace mindspore
