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
#include "kernel/ascend/opapi/aclnn/avg_pool2d_grad_aclnn_kernel.h"

#include <tuple>
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>

#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace {
std::tuple<std::vector<int64_t>, std::vector<int64_t>, std::vector<int64_t>, std::tuple<bool, bool, int64_t, int8_t>>
AvgPool2DGradGenerate(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  auto kernel_size = inputs[kIndex2]->GetValueWithCheck<std::vector<int64_t>>();
  auto stride = inputs[kIndex3]->GetValueWithCheck<std::vector<int64_t>>();
  auto padding = inputs[kIndex4]->GetValueWithCheck<std::vector<int64_t>>();
  bool ceil_mode = inputs[kIndex5]->GetValueWithCheck<bool>();
  bool count_include_pad = inputs[kIndex6]->GetValueWithCheck<bool>();
  int64_t divisor_override = 0;
  if (inputs[kIndex7]->GetType()->type_id() != kMetaTypeNone) {
    divisor_override = inputs[kIndex7]->GetValueWithCheck<int64_t>();
  }
  int8_t cube_math_type = OpApiUtil::GetCubeMathType();
  return std::make_tuple(std::move(kernel_size), std::move(stride), std::move(padding),
                         std::make_tuple(ceil_mode, count_include_pad, divisor_override, cube_math_type));
}
}  // namespace

void AvgPool2DGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  auto params = AvgPool2DGradGenerate(inputs, outputs);
  const auto &kernel_size = std::get<0>(params);
  const auto &stride = std::get<1>(params);
  const auto &padding = std::get<2>(params);
  auto [ceil_mode, count_include_pad, divisor_override, cube_math_type] = std::get<3>(params);
  GetWorkspaceForResize(inputs[0], inputs[1], kernel_size, stride, padding, ceil_mode, count_include_pad,
                        divisor_override, cube_math_type, outputs[0]);
}

bool AvgPool2DGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  auto params = AvgPool2DGradGenerate(inputs, outputs);
  const auto &kernel_size = std::get<0>(params);
  const auto &stride = std::get<1>(params);
  const auto &padding = std::get<2>(params);
  auto [ceil_mode, count_include_pad, divisor_override, cube_math_type] = std::get<3>(params);
  RunOp(stream_ptr, workspace, inputs[0], inputs[1], kernel_size, stride, padding, ceil_mode, count_include_pad,
        divisor_override, cube_math_type, outputs[0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AvgPool2DGrad, AvgPool2DGradAscend);
}  // namespace kernel
}  // namespace mindspore
