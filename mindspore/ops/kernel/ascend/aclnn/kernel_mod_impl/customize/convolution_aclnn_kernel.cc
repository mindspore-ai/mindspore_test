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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/convolution_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace convolution {
namespace {
void ExpandParamIfNeeded(std::vector<int64_t> *const param, size_t expect_dim) {
  if (param->size() == kIndex1) {
    param->insert(param->end(), expect_dim - kIndex1, param->at(kIndex0));
  }
}
}  // namespace
void ConvolutionAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  const auto &weight_shape = inputs[kIndex1]->GetShapeVector();
  auto spatial_len = weight_shape.size() - kIndex2;
  stride_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex3]);
  ExpandParamIfNeeded(&stride_, spatial_len);
  padding_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex4]);
  ExpandParamIfNeeded(&padding_, spatial_len);
  dilation_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex5]);
  ExpandParamIfNeeded(&dilation_, spatial_len);
  transposed_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex6]);
  output_padding_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex7]);
  ExpandParamIfNeeded(&output_padding_, spatial_len);
  groups_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex8]);
  cube_math_type_ = OpApiUtil::GetCubeMathType(OpApiUtil::IsAllowConvHF32());

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], stride_, padding_, dilation_, transposed_,
                        output_padding_, groups_, outputs[kIndex0], cube_math_type_);
}

bool ConvolutionAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], stride_, padding_, dilation_,
        transposed_, output_padding_, groups_, outputs[kIndex0], cube_math_type_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Convolution, ConvolutionAscend);
}  // namespace convolution
}  // namespace kernel
}  // namespace mindspore
