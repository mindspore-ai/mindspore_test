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

#include "kernel/ascend/opapi/aclnn/full_like_aclnn_kernel.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {

void FullLikeAscendCall::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  fill_value_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex1]);
  GetWorkspaceForResize(outputs[kIndex0], fill_value_);
}

bool FullLikeAscendCall::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, outputs[kIndex0], fill_value_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(FullLike, FullLikeAscendCall);
}  // namespace kernel
}  // namespace mindspore
