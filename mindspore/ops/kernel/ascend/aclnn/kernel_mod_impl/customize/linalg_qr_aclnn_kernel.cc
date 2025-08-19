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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/linalg_qr_aclnn_kernel.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace linalg_qr {
void LinalgQrAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  mode_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  GetWorkspaceForResize(inputs[kIndex0], mode_, outputs[kIndex0], outputs[kIndex1]);
}

bool LinalgQrAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  RunOp(stream_ptr, workspace, inputs[kIndex0], mode_, outputs[kIndex0], outputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(LinalgQr, LinalgQrAclnnKernelMod);
}  // namespace linalg_qr
}  // namespace kernel
}  // namespace mindspore
