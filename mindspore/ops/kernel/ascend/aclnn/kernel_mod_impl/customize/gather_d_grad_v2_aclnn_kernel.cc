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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/gather_d_grad_v2_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <string>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace gather_d_grad_v2 {

void GatherDGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  dim_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  ClearOpsWorkSpaceList();
  GetWorkspaceForResizeInplaceZero(outputs[kIndex0]);
  GetWorkspaceForResizeScatterAdd(inputs[kIndex0], dim_, inputs[kIndex2], inputs[kIndex3], outputs[kIndex0]);
}

bool GatherDGradAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOpInplaceZero(stream_ptr, workspace, outputs[kIndex0]);
  RunOpScatterAdd(stream_ptr, workspace, outputs[kIndex0], dim_, inputs[kIndex2], inputs[kIndex3], outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(GatherDGradV2, GatherDGradAscend);
}  // namespace gather_d_grad_v2
}  // namespace kernel
}  // namespace mindspore
