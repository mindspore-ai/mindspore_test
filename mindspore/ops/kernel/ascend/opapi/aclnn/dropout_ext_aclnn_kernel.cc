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
#include "kernel/ascend/opapi/aclnn/dropout_ext_aclnn_kernel.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "mindspore/core/include/mindapi/base/types.h"

namespace mindspore {
namespace kernel {
namespace dropout_ext {
void DropoutExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  MS_EXCEPTION_IF_NULL(primitive_);
  p_value_ = inputs[kIndex1]->GetValueWithCheck<pyfloat>();
  seed_value_ = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  offset_value_ = inputs[kIndex3]->GetValueWithCheck<int64_t>();
  dtype_value_ = inputs[kIndex0]->dtype_id();
  GetWorkspaceForResizeGenMask(inputs[kIndex0]->GetShapeVector(), p_value_, seed_value_, offset_value_, dtype_value_,
                               outputs[kIndex1]);
  GetWorkspaceForResizeDoMask(inputs[kIndex0], outputs[kIndex1], p_value_, outputs[kIndex0]);
}

bool DropoutExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOpGenMask(stream_ptr, workspace, inputs[kIndex0]->GetShapeVector(), p_value_, seed_value_, offset_value_,
               dtype_value_, outputs[kIndex1]);

  RunOpDoMask(stream_ptr, workspace, inputs[kIndex0], outputs[kIndex1], p_value_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(DropoutExt, DropoutExtAscend);
}  // namespace dropout_ext
}  // namespace kernel
}  // namespace mindspore
