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
#include "kernel/ascend/opapi/aclnn/cross_aclnn_kernel.h"
#include "ir/tensor.h"
#include "infer/ops_func_impl/cross.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
namespace cross {

void CrossAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  dim_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  const int64_t default_dim = -65530;
  if (dim_ == default_dim) {
    const auto &input_shape = inputs[kIndex0]->GetShape()->GetShapeVector();
    const auto &other_shape = inputs[kIndex1]->GetShape()->GetShapeVector();
    dim_ = SizeToLong(ops::CalCrossDimFromDefaultValue(input_shape, other_shape));
  }
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], dim_, outputs[kIndex0]);
}

bool CrossAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], dim_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Cross, CrossAscend);
}  // namespace cross
}  // namespace kernel
}  // namespace mindspore
