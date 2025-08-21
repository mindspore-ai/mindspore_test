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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/nllloss_grad_aclnn_kernel.h"
#include <vector>
#include <unordered_map>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {
namespace nllloss_grad {
void NLLLossGradAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  auto reduction_imm = static_cast<Reduction>(device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex5]));
  auto reduction = device::ascend::AclHelper::ConvertMsReductionToGe(reduction_imm);

  auto ignore_index = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex6]);

  GetWorkspaceForResize(inputs[kIndex1], inputs[kIndex0], inputs[kIndex2], inputs[kIndex3], reduction, ignore_index,
                        inputs[kIndex4], outputs[kIndex0]);
}

bool NLLLossGradAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto reduction_imm = static_cast<Reduction>(device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex5]));
  auto reduction = device::ascend::AclHelper::ConvertMsReductionToGe(reduction_imm);

  auto ignore_index = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex6]);

  RunOp(stream_ptr, workspace, inputs[kIndex1], inputs[kIndex0], inputs[kIndex2], inputs[kIndex3], reduction,
        ignore_index, inputs[kIndex4], outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(NLLLossGrad, NLLLossGradAclnnKernelMod);
}  // namespace nllloss_grad
}  // namespace kernel
}  // namespace mindspore
