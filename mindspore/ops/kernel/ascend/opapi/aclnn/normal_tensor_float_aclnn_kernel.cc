/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/opapi/aclnn/normal_tensor_float_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/value_utils.h"

namespace mindspore {
namespace kernel {
namespace normal_tensor_float {

void NormalTensorFloatAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  auto std_scalar = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex1]);
  std_ = GetScalarCastValue<float>("NormalTensorFloat", std_scalar);
  seed_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  offset_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  GetWorkspaceForResize(inputs[kIndex0], std_, seed_, offset_, outputs[kIndex0]);
}

bool NormalTensorFloatAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &workspace,
                                     const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  RunOp(stream_ptr, workspace, inputs[kIndex0], std_, seed_, offset_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(NormalTensorFloat, NormalTensorFloatAscend);
}  // namespace normal_tensor_float
}  // namespace kernel
}  // namespace mindspore
