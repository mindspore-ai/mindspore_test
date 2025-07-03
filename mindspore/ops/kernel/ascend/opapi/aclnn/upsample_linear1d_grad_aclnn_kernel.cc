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
#include "kernel/ascend/opapi/aclnn/upsample_linear1d_grad_aclnn_kernel.h"

#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include <tuple>

#include "ir/tensor.h"
#include "mindapi/base/types.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace upsample_linear1d_grad {
namespace {
const pyfloat DEFAULT_SCALE_VALUE = -1;
std::tuple<std::vector<int64_t>, std::vector<int64_t>, double, bool> UpsampleLinear1DGradGenerate(
  const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  bool align_corners = inputs[kIndex4]->GetValueWithCheck<bool>();
  auto input_size = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();

  auto grad_out_shape = inputs[kIndex0]->GetShapeVector();
  std::vector<int64_t> output_size{grad_out_shape.begin() + kIndex2, grad_out_shape.end()};

  std::vector<pyfloat> scales{DEFAULT_SCALE_VALUE};
  if (inputs[kIndex3]->GetType()->type_id() != kMetaTypeNone) {
    if (!align_corners) {
      MS_LOG(EXCEPTION) << "For UpsampleLinear1DGrad with align_corners false, scales was not supported.";
    }
    scales = inputs[kIndex3]->GetValueWithCheck<std::vector<pyfloat>>();
  }

  MS_ASSERT(scales.size() == kIndex1);
  double scales_l = scales[0];

  return std::make_tuple(std::move(input_size), std::move(output_size), scales_l, align_corners);
}
}  // namespace

void UpsampleLinear1DGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  auto params = UpsampleLinear1DGradGenerate(inputs, outputs);
  input_size_ = std::get<kIndex0>(params);
  output_size_ = std::get<kIndex1>(params);
  scales_l_ = std::get<kIndex2>(params);
  align_corners_ = std::get<kIndex3>(params);
  GetWorkspaceForResize(inputs[kIndex0], output_size_, input_size_, align_corners_, scales_l_, outputs[kIndex0]);
}

bool UpsampleLinear1DGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], output_size_, input_size_, align_corners_, scales_l_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(UpsampleLinear1DGrad, UpsampleLinear1DGradAscend);
}  // namespace upsample_linear1d_grad
}  // namespace kernel
}  // namespace mindspore
