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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/adaptive_max_pool2d_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace adaptive_max_pool2d {
namespace {
constexpr int kShapeDim2d = 2;
constexpr int kShapeDimNone = -1;
}  // namespace

void AdaptiveMaxPool2DAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  auto output_size = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  if (output_size.size() != 2) {
    MS_LOG(EXCEPTION) << "For AdaptiveMaxPool2D, the output_size size should be 2.";
  }
  auto input_shape = inputs[kIndex0]->GetShapeVector();
  auto input_shape_size = input_shape.size();

  output_size_vector_ = {};
  for (auto i = 0; i < kShapeDim2d; i++) {
    if (output_size[i] != kShapeDimNone) {
      output_size_vector_.emplace_back(output_size[i]);
    } else {
      output_size_vector_.emplace_back(input_shape[input_shape_size - kShapeDim2d + i]);
    }
  }

  GetWorkspaceForResizeAdaptiveMaxPool2D(inputs[kIndex0], output_size_vector_, outputs[kIndex0], outputs[kIndex1]);
}

bool AdaptiveMaxPool2DAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOpAdaptiveMaxPool2D(stream_ptr, workspace, inputs[kIndex0], output_size_vector_, outputs[kIndex0],
                         outputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(AdaptiveMaxPool2D, AdaptiveMaxPool2DAscend);
}  // namespace adaptive_max_pool2d
}  // namespace kernel
}  // namespace mindspore
