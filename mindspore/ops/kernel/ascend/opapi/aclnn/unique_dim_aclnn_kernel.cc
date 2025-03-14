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
#include "kernel/ascend/opapi/aclnn/unique_dim_aclnn_kernel.h"
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
namespace unique_dim {

void UniqueDimAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  auto sorted = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex1]);
  auto return_inverse = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex2]);
  auto dim = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  GetWorkspaceForResize(inputs[kIndex0], sorted, return_inverse, dim, outputs[kIndex0], outputs[kIndex1],
                        outputs[kIndex2]);
}

bool UniqueDimAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(DEBUG) << "Run UniqueDim start.";

  auto sorted = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex1]);
  auto return_inverse = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex2]);
  auto dim = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]);

  auto res = GEN_EXECUTOR_CUST(op_type_, inputs[kIndex0], sorted, return_inverse, dim, outputs[kIndex0],
                               outputs[kIndex1], outputs[kIndex2]);
  UpdateWorkspace(res);
  executor_ = std::get<kIndex1>(res);
  auto &all_acl_tensor = std::get<kIndex2>(res);
  RunOpSync(stream_ptr, workspace);
  MS_LOG(DEBUG) << "Run UniqueDim end.";

  // update output shape
  size_t output_size = 3;
  output_shapes_.resize(output_size);
  output_shapes_[kIndex0] = device::ascend::UpdateOutputShape(all_acl_tensor.get<kIndex4>());
  output_shapes_[kIndex1] = device::ascend::UpdateOutputShape(all_acl_tensor.get<kIndex5>());
  output_shapes_[kIndex2] = device::ascend::UpdateOutputShape(all_acl_tensor.get<kIndex6>());
  return true;
}

void UniqueDimAscend::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                               const std::vector<KernelTensor *> &outputs) {
  for (size_t i = 0; i < output_shapes_.size(); ++i) {
    outputs[i]->SetShapeVector(output_shapes_[i]);
    size_t dtype_byte = GetTypeByte(TypeIdToType(outputs[i]->dtype_id()));
    size_t update_size = LongToSize(
      std::accumulate(output_shapes_[i].begin(), output_shapes_[i].end(), dtype_byte, std::multiplies<int64_t>()));
    outputs[i]->set_size(update_size);
  }
}

MS_ACLNN_KERNEL_FACTORY_REG(UniqueDim, UniqueDimAscend);
}  // namespace unique_dim
}  // namespace kernel
}  // namespace mindspore
