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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/argsort_aclnn_kernel.h"

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
namespace argsort {

void ArgSortAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  auto input_tensor = inputs[kIndex0];
  const auto &input_shape = input_tensor->GetShapeVector();
  output_kernel_tensor_.SetType(input_tensor->GetType());
  output_kernel_tensor_.SetShape(std::make_shared<abstract::TensorShape>(input_shape));

  dim = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex1]);
  descending = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex2]);
  stable = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex3]);

  GetWorkspaceForResizeSort(input_tensor, stable, dim, descending, &output_kernel_tensor_, outputs[kIndex0]);

  const auto &output_size = ops::CalOutputSize(output_kernel_tensor_.GetShapeVector(),
                                               mindspore::abstract::TypeIdSize(output_kernel_tensor_.dtype_id()));
  workspace_size_list_.emplace_back(output_size);
}

bool ArgSortAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  size_t workspace_offset = LongToSize(SizeToLong(workspace.size()) - SizeToLong(1));
  output_kernel_tensor_.set_device_ptr(workspace[workspace_offset]->device_ptr());
  RunOpSort(stream_ptr, workspace, inputs[kIndex0], stable, dim, descending, &output_kernel_tensor_, outputs[kIndex0]);

  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(ArgSort, ArgSortAscend);
}  // namespace argsort
}  // namespace kernel
}  // namespace mindspore
