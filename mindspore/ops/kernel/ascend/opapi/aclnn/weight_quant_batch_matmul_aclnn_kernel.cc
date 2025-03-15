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
#include "kernel/ascend/opapi/aclnn/weight_quant_batch_matmul_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace weight_quant_batch_matmul {

void WeightQuantBatchMatmulV2Ascend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  auto trans_x = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex7]);
  auto trans_weight = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex8]);
  antiquant_group_size_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex9]);

  weight_ = std::make_shared<KernelTensor>(*inputs[kIndex1]);
  KernelTensor *weight = weight_.get();
  if (weight->dtype_id() == kNumberTypeInt4) {
    ShapeVector weight_shape = weight->GetShapeVector();
    int kInt4ShapeMul = 2;
    weight_shape.back() *= kInt4ShapeMul;
    weight->SetShapeVector(weight_shape);
  }

  input_x_ = std::pair<KernelTensor *, bool>(inputs[kIndex0], trans_x);
  input_weight_ = std::pair<KernelTensor *, bool>(weight, trans_weight);
  GetWorkspaceForResize(input_x_, input_weight_, inputs[kIndex2], inputs[kIndex3], inputs[kIndex4], inputs[kIndex5],
                        inputs[kIndex6], antiquant_group_size_, outputs[kIndex0]);
}

bool WeightQuantBatchMatmulV2Ascend::Launch(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  input_x_.first = inputs[kIndex0];

  if (inputs[kIndex1]->dtype_id() == kNumberTypeInt4) {
    weight_ = std::make_shared<KernelTensor>(*inputs[kIndex1]);
    KernelTensor *weight = weight_.get();
    ShapeVector weight_shape = weight->GetShapeVector();
    int kInt4ShapeMul = 2;
    weight_shape.back() *= kInt4ShapeMul;
    weight->SetShapeVector(weight_shape);
    input_weight_.first = weight;
  } else {
    input_weight_.first = inputs[kIndex1];
  }

  RunOp(stream_ptr, workspace, input_x_, input_weight_, inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
        inputs[kIndex5], inputs[kIndex6], antiquant_group_size_, outputs[kIndex0]);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(WeightQuantBatchMatmul, WeightQuantBatchMatmulV2Ascend);
}  // namespace weight_quant_batch_matmul
}  // namespace kernel
}  // namespace mindspore
