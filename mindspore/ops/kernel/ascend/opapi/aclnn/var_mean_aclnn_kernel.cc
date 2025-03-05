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
#include "kernel/ascend/opapi/aclnn/var_mean_aclnn_kernel.h"
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

void VarMeanAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  const auto dim_opt = inputs[kIndex1]->GetOptionalValueWithCheck<std::vector<int64_t>>();
  if (dim_opt.has_value()) {
    dim_ = dim_opt.value();
  } else {
    dim_ = std::vector<int64_t>{};
  }
  correction_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  keepdim_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex3]);

  GetWorkspaceForResize(inputs[kIndex0], dim_, correction_, keepdim_, outputs[kIndex0], outputs[kIndex1]);
}

bool VarMeanAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], dim_, correction_, keepdim_, outputs[kIndex0], outputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(VarMean, VarMeanAscend);
}  // namespace kernel
}  // namespace mindspore
