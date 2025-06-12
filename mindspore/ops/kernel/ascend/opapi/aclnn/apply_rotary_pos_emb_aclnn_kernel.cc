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

#include "kernel/ascend/opapi/aclnn/apply_rotary_pos_emb_aclnn_kernel.h"
#include <vector>
#include <string>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
void ApplyRotaryPosEmbAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], bsnd_layout_);
  return;
}

bool ApplyRotaryPosEmbAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], bsnd_layout_);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(ApplyRotaryPosEmbExt, ApplyRotaryPosEmbAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
