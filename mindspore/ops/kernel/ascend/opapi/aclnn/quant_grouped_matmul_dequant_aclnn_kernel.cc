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
#include "kernel/ascend/opapi/aclnn/quant_grouped_matmul_dequant_aclnn_kernel.h"
#include <algorithm>
#include <string>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "plugin/res_manager/ascend/op_adapter/op_adapter_base.h"

namespace mindspore {
namespace kernel {
namespace quant_grouped_matmul_dequant {
namespace {
constexpr size_t kXIndex = 0;
constexpr size_t kWeightIndex = 1;
constexpr size_t kWeightScaleIndex = 2;
constexpr size_t kGroupListIndex = 3;
constexpr size_t kBiasIndex = 4;
constexpr size_t kXScaleIndex = 5;
constexpr size_t kXOffsetIndex = 6;
constexpr size_t kSmoothScaleIndex = 7;
constexpr size_t kQuantModeIndex = 8;
constexpr size_t kTransposeWeightIndex = 9;
}  // namespace

void QuantGroupedMatmulDequantAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                       const std::vector<KernelTensor *> &outputs) {
  auto quant_mode = inputs[kQuantModeIndex]->GetOptionalValueWithCheck<std::string>();
  quant_mode_ = quant_mode.has_value() ? quant_mode.value() : "pertoken";
  auto transpose_weight = inputs[kTransposeWeightIndex]->GetOptionalValueWithCheck<bool>();
  transpose_weight_ = transpose_weight.has_value() ? transpose_weight.value() : true;

  GetWorkspaceForResize(inputs[kXIndex], inputs[kWeightIndex], inputs[kWeightScaleIndex], inputs[kGroupListIndex],
                        inputs[kBiasIndex], inputs[kXScaleIndex], inputs[kXOffsetIndex], inputs[kSmoothScaleIndex],
                        quant_mode_, transpose_weight_, outputs[kIndex0]);
}

bool QuantGroupedMatmulDequantAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &workspace,
                                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kXIndex], inputs[kWeightIndex], inputs[kWeightScaleIndex],
        inputs[kGroupListIndex], inputs[kBiasIndex], inputs[kXScaleIndex], inputs[kXOffsetIndex],
        inputs[kSmoothScaleIndex], quant_mode_, transpose_weight_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(QuantGroupedMatmulDequant, QuantGroupedMatmulDequantAscend);
}  // namespace quant_grouped_matmul_dequant
}  // namespace kernel
}  // namespace mindspore
