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
#include "mindspore/ops/kernel/ascend/opapi/aclnn/pow_tensor_scalar_aclnn_kernel.h"

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
namespace pow_tensor_scalar {

void PowTensorScalarAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  auto exponent = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex1]);

  GetWorkspaceForResize(inputs[kIndex0], exponent, outputs[kIndex0]);
}

bool PowTensorScalarAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto exponent = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex1]);

  RunOp(stream_ptr, workspace, inputs[kIndex0], exponent, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(PowTensorScalar, PowTensorScalarAscend);
}  // namespace pow_tensor_scalar
}  // namespace kernel
}  // namespace mindspore
