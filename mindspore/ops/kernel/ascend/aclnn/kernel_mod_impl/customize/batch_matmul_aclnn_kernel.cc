/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/batch_matmul_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace batch_matmul {
void BMMAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  bool trans_a = inputs[kIndex2]->GetValueWithCheck<bool>();
  bool trans_b = inputs[kIndex3]->GetValueWithCheck<bool>();

  input_a_ = std::pair<KernelTensor *, bool>(inputs[kIndex0], trans_a);
  input_b_ = std::pair<KernelTensor *, bool>(inputs[kIndex1], trans_b);
  cube_math_type_ = OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowMatmulHF32());

  GetWorkspaceForResize(input_a_, input_b_, outputs[kIndex0], cube_math_type_);
}

bool BMMAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  input_a_.first = inputs[kIndex0];
  input_b_.first = inputs[kIndex1];
  RunOp(stream_ptr, workspace, input_a_, input_b_, outputs[kIndex0], cube_math_type_);
  return true;
}

void BMMExtAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &outputs) {
  cube_math_type_ = OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowMatmulHF32());
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], outputs[kIndex0], cube_math_type_);
}

bool BMMExtAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &workspace,
                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], outputs[kIndex0], cube_math_type_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(BatchMatMulExt, BMMExtAclnnKernelMod);
MS_ACLNN_KERNEL_FACTORY_REG(BatchMatMul, BMMAclnnKernelMod);
}  // namespace batch_matmul
}  // namespace kernel
}  // namespace mindspore
