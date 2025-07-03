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
#include "kernel/ascend/opapi/aclnn/upsample_trilinear3d_grad_aclnn_kernel.h"

#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <tuple>
#include <functional>

#include "ir/tensor.h"
#include "mindapi/base/types.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace upsample_trilinear3d_grad {
namespace {
std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::tuple<double, double, double>, bool>
UpsampleTrilinear3DGradGenerate(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto input_size = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();

  auto grad_out_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> output_size{grad_out_shape.begin() + kIndex2, grad_out_shape.end()};

  std::vector<pyfloat> scales{0., 0., 0.};
  if (inputs[kIndex3]->GetType()->type_id() != kMetaTypeNone) {
    scales = inputs[kIndex3]->GetValueWithCheck<std::vector<pyfloat>>();
  }

  bool align_corners = inputs[kIndex4]->GetValueWithCheck<bool>();

  MS_ASSERT(scales.size() == kIndex3);
  double scales_d = scales[kIndex0];
  double scales_h = scales[kIndex1];
  double scales_w = scales[kIndex2];

  return std::make_tuple(std::move(input_size), std::move(output_size), std::make_tuple(scales_d, scales_h, scales_w),
                         align_corners);
}
}  // namespace

void UpsampleTrilinear3DGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  auto params = UpsampleTrilinear3DGradGenerate(inputs, outputs);
  input_size_ = std::get<kIndex0>(params);
  output_size_ = std::get<kIndex1>(params);
  std::tie(scales_d_, scales_h_, scales_w_) = std::get<kIndex2>(params);
  align_corners_ = std::get<kIndex3>(params);
  GetWorkspaceForResize(inputs[kIndex0], output_size_, input_size_, align_corners_, scales_d_, scales_h_, scales_w_,
                        outputs[kIndex0]);
}

bool UpsampleTrilinear3DGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], output_size_, input_size_, align_corners_, scales_d_, scales_h_,
        scales_w_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(UpsampleTrilinear3DGrad, UpsampleTrilinear3DGradAscend);
}  // namespace upsample_trilinear3d_grad
}  // namespace kernel
}  // namespace mindspore
