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
#include "mindspore/ops/kernel/ascend/opapi/aclnn/inplace_masked_scatter_aclnn_kernel.h"
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

void InplaceMaskedScatterAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  auto input_dtype_id = inputs[kIndex0]->dtype_id();
  if (input_dtype_id == kNumberTypeFloat64 || input_dtype_id == kNumberTypeInt16) {
    MS_EXCEPTION(ValueError) << "For InplaceMaskedScatter, the type of 'input' is no support Tensor[Float64, Int16] ";
  }
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2]);
}

bool InplaceMaskedScatterAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(InplaceMaskedScatter, InplaceMaskedScatterAscend);
}  // namespace kernel
}  // namespace mindspore
