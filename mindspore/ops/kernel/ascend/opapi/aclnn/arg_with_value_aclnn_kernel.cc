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
#include "kernel/ascend/opapi/aclnn/arg_with_value_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace arg_with_value {

void ArgMaxWithValueAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  axis_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  keep_dims_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex2]);
  GetWorkspaceForResize(inputs[kIndex0], axis_, keep_dims_, outputs[kIndex1], outputs[kIndex0]);
}

bool ArgMaxWithValueAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], axis_, keep_dims_, outputs[kIndex1], outputs[kIndex0]);
  return true;
}

void ArgMinWithValueAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  axis_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  keep_dims_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex2]);
  GetWorkspaceForResize(inputs[kIndex0], axis_, keep_dims_, outputs[kIndex1], outputs[kIndex0]);
}

bool ArgMinWithValueAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], axis_, keep_dims_, outputs[kIndex1], outputs[kIndex0]);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(ArgMaxWithValue, ArgMaxWithValueAscend);
MS_ACLNN_KERNEL_FACTORY_REG(ArgMinWithValue, ArgMinWithValueAscend);
}  // namespace arg_with_value
}  // namespace kernel
}  // namespace mindspore
