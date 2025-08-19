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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/fill_tensor_aclnn_kernel.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace fill_tensor {

void FillTensorAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  auto value_tensor = inputs[kIndex1];
  if (value_tensor->device_ptr() == nullptr) {
    MS_LOG(INFO) << "For " << primitive_->name() << ", Input [fill_value] is a host tensor, FillScalar will be used.";
    value_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex1]);
    op_type_ = "aclnnInplaceFillScalar";
    GetWorkspaceForResize(outputs[kIndex0], value_);
    return;
  }
  GetWorkspaceForResize(outputs[kIndex0], inputs[kIndex1]);
}

bool FillTensorAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (op_type_ == "aclnnInplaceFillScalar") {
    RunOp(stream_ptr, workspace, outputs[kIndex0], value_);
  } else {
    RunOp(stream_ptr, workspace, outputs[kIndex0], inputs[kIndex1]);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(FillTensor, FillTensorAscend);
}  // namespace fill_tensor
}  // namespace kernel
}  // namespace mindspore
