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
#include "mindspore/ops/kernel/ascend/opapi/aclnn/rotary_position_embedding_grad_aclnn_kernel.h"
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
namespace rotary_position_embedding_grad {

void RotaryPositionEmbeddingGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                         const std::vector<KernelTensor *> &outputs) {
  auto mode = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex4]);

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], mode, outputs[kIndex0],
                        outputs[kIndex1], outputs[kIndex2]);
}

bool RotaryPositionEmbeddingGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &workspace,
                                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto mode = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex4]);

  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], mode,
        outputs[kIndex0], outputs[kIndex1], outputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(RotaryPositionEmbeddingGrad, RotaryPositionEmbeddingGradAscend);
}  // namespace rotary_position_embedding_grad
}  // namespace kernel
}  // namespace mindspore
