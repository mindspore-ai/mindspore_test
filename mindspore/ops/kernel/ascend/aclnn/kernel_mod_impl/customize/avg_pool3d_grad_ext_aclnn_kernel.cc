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
#include "mindspore/ops/kernel/ascend/aclnn/kernel_mod_impl/customize/avg_pool3d_grad_ext_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace avg_pool3d_grad_ext {

void AvgPool3dGradExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  auto kernel_size = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex2]);
  auto stride = inputs[kIndex3]->GetType()->type_id() != kMetaTypeNone
                  ? device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex3])
                  : kernel_size;
  auto padding = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex4]);
  auto ceil_mode = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex5]);
  auto count_include_pad = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex6]);
  int64_t divisor_override = 0;
  if (inputs[kIndex7]->GetType()->type_id() != kMetaTypeNone) {
    divisor_override = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex7]);
  }

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], kernel_size, stride, padding, ceil_mode, count_include_pad,
                        divisor_override, outputs[kIndex0]);
}

bool AvgPool3dGradExtAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto kernel_size = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex2]);
  auto stride = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex3]);
  auto padding = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex4]);
  auto ceil_mode = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex5]);
  auto count_include_pad = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex6]);
  int64_t divisor_override = 0;
  if (inputs[kIndex7]->GetType()->type_id() != kMetaTypeNone) {
    divisor_override = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex7]);
  }

  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], kernel_size, stride, padding, ceil_mode,
        count_include_pad, divisor_override, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AvgPool3DGradExt, AvgPool3dGradExtAscend);
}  // namespace avg_pool3d_grad_ext
}  // namespace kernel
}  // namespace mindspore
