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
#include "kernel/ascend/opapi/aclnn/inplace_threshold_aclnn_kernel.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace inplace_threshold {

void InplaceThresholdAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  threshold_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex1]);
  value_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex2]);
  GetWorkspaceForResize(inputs[kIndex0], threshold_, value_);
}

bool InplaceThresholdAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], threshold_, value_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(InplaceThreshold, InplaceThresholdAscend);
}  // namespace inplace_threshold
}  // namespace kernel
}  // namespace mindspore
