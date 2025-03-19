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
#include "kernel/ascend/opapi/aclnn/roi_align_grad_ext_aclnn_kernel.h"
#include <vector>
#include <unordered_map>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {
void RoiAlignGradExtAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  input_shape_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex2]);
  output_size_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex3]);
  pooled_height_ = output_size_[kDim0];
  if (output_size_.size() == kDim2) {
    pooled_width_ = output_size_[kDim1];
  } else {
    pooled_width_ = output_size_[kDim0];
  }
  spatial_scale_ = device::ascend::ConvertKernelTensor<float>(inputs[kIndex4]);
  sampling_ratio_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex5]);
  aligned_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex6]);

  if (sampling_ratio_ < 0) {
    sampling_ratio_ = 0;
  }

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], input_shape_, pooled_height_, pooled_width_, spatial_scale_,
                        sampling_ratio_, aligned_, outputs[kIndex0]);
}

bool RoiAlignGradExtAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], input_shape_, pooled_height_, pooled_width_,
        spatial_scale_, sampling_ratio_, aligned_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(RoiAlignGradExt, RoiAlignGradExtAclnnKernelMod);
}  // namespace kernel
}  // namespace mindspore
