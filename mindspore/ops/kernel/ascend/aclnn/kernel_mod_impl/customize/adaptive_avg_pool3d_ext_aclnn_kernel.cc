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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/adaptive_avg_pool3d_ext_aclnn_kernel.h"
#include "ir/tensor.h"

namespace mindspore {
namespace kernel {
namespace adaptive_avg_pool3d_ext {
namespace {
constexpr int kShapeDim1d = 1;
constexpr int kShapeDim3d = 3;
constexpr int kShapeDimNone = -1;
}  // namespace
void AdaptiveAvgPool3DExtAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                          const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  auto output_size = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  auto input_shape_size = input_shape.size();

  output_size_vector_ = {};
  for (auto i = 0; i < kShapeDim3d; i++) {
    if (output_size[i] != kShapeDimNone)
      output_size_vector_.emplace_back(output_size[i]);
    else
      output_size_vector_.emplace_back(input_shape[input_shape_size - kShapeDim3d + i]);
  }
  dtype_ = outputs[kIndex0]->dtype_id();
  if (output_size_vector_[kIndex0] == kShapeDim1d && output_size_vector_[kIndex1] == kShapeDim1d &&
      output_size_vector_[kIndex2] == kShapeDim1d) {
    GetWorkspaceForResizeMeanExt(inputs[kIndex0], axis_, keep_dims_, dtype_, outputs[kIndex0]);
  } else {
    GetWorkspaceForResizeAdaptiveAvgPool3DExt(inputs[kIndex0], output_size_vector_, outputs[kIndex0]);
  }
}

bool AdaptiveAvgPool3DExtAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                                const std::vector<KernelTensor *> &workspace,
                                                const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (output_size_vector_[kIndex0] == kShapeDim1d && output_size_vector_[kIndex1] == kShapeDim1d &&
      output_size_vector_[kIndex2] == kShapeDim1d) {
    RunOpMeanExt(stream_ptr, workspace, inputs[kIndex0], axis_, keep_dims_, dtype_, outputs[kIndex0]);
  } else {
    RunOpAdaptiveAvgPool3DExt(stream_ptr, workspace, inputs[kIndex0], output_size_vector_, outputs[kIndex0]);
  }
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(AdaptiveAvgPool3DExt, AdaptiveAvgPool3DExtAclnnKernelMod);
}  // namespace adaptive_avg_pool3d_ext
}  // namespace kernel
}  // namespace mindspore
