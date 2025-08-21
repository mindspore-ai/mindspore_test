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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/triangular_solve_aclnn_kernel.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace triangular_solve {

void TriangularSolveAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  upper_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex2]);
  transpose_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex3]);
  unitriangular_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex4]);
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], upper_, transpose_, unitriangular_, outputs[kIndex0],
                        outputs[kIndex1]);
}

bool TriangularSolveAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], upper_, transpose_, unitriangular_, outputs[kIndex0],
        outputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(TriangularSolve, TriangularSolveAscend);
}  // namespace triangular_solve
}  // namespace kernel
}  // namespace mindspore
