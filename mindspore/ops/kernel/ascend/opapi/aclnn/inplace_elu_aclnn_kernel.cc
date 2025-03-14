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
#include "kernel/ascend/opapi/aclnn/inplace_elu_aclnn_kernel.h"
#include <vector>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace inplace_elu {
void InplaceEluAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &outputs) {
  TypeId data_type = inputs[kIndex0]->dtype_id();
  if (data_type == kNumberTypeFloat64) {
    MS_LOG(EXCEPTION) << "Unsupported input dtype: float64, because aclnnEluBackward does not support dtype: float64";
  }
  alpha_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex1]);
  MAKE_SCALAR(1.f, inputs[kIndex0]->dtype_id(), scale_);
  MAKE_SCALAR(1.f, inputs[kIndex0]->dtype_id(), input_scale_);

  GetWorkspaceForResize(inputs[kIndex0], alpha_, scale_, input_scale_);
}

bool InplaceEluAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &workspace,
                                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], alpha_, scale_, input_scale_);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(InplaceElu, InplaceEluAclnnKernelMod);
}  // namespace inplace_elu
}  // namespace kernel
}  // namespace mindspore
