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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/lerp_scalar_aclnn_kernel.h"
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace lerp_scalar {

void LerpScalarAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  auto weight_dtype_id = inputs[kIndex2]->dtype_id();
  auto weight = inputs[kIndex2]->GetValueWithCheck<float>();
  MAKE_SCALAR(weight, weight_dtype_id, weight_);
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], weight_, outputs[kIndex0]);
}

bool LerpScalarAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], weight_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(LerpScalar, LerpScalarAscend);
}  // namespace lerp_scalar
}  // namespace kernel
}  // namespace mindspore
