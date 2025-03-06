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
#include "kernel/ascend/opapi/aclnn/inplace_hardtanh_aclnn_kernel.h"
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

void InplaceHardtanhAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  min_val_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex1]);
  max_val_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex2]);

  GetWorkspaceForResize(inputs[kIndex0], min_val_, max_val_);
}

bool InplaceHardtanhAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], min_val_, max_val_);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(InplaceHardtanh, InplaceHardtanhAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
