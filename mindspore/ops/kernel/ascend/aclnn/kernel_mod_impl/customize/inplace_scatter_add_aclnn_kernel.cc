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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/inplace_scatter_add_aclnn_kernel.h"
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace inplace_scatter_add {

void InplaceScatterAddAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  dim_ = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  GetWorkspaceForResize(inputs[kIndex0], dim_, inputs[kIndex2], inputs[kIndex3], inputs[kIndex0]);
}

bool InplaceScatterAddAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], dim_, inputs[kIndex2], inputs[kIndex3], inputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(InplaceScatterAdd, InplaceScatterAddAclnnKernelMod);
}  // namespace inplace_scatter_add
}  // namespace kernel
}  // namespace mindspore
