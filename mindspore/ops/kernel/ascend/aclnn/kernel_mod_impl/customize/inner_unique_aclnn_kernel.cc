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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/inner_unique_aclnn_kernel.h"
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"

namespace mindspore {
namespace kernel {
namespace inner_unique {
void InnerUniqueAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  sorted_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex1]);
  return_inverse_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex2]);
  GetWorkspaceForResize(inputs[kIndex0], sorted_, return_inverse_, outputs[kIndex0], outputs[kIndex1]);
}

bool InnerUniqueAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                               const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  MS_LOG(DEBUG) << "Run InnerUnique start.";
  const auto &all_acl_tensor =
    RunOpSync(stream_ptr, workspace, inputs[kIndex0], sorted_, return_inverse_, outputs[kIndex0], outputs[kIndex1]);
  MS_LOG(DEBUG) << "Run InnerUnique end.";

  // update output shape
  auto output_real_shapes = ShapeArray{all_acl_tensor.at(kIndex3), all_acl_tensor.at(kIndex4)};
  output_shapes_ = std::move(output_real_shapes);
  return true;
}

void InnerUniqueAscend::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  for (size_t i = 0; i < output_shapes_.size(); ++i) {
    outputs[i]->SetShapeVector(output_shapes_[i]);
    size_t dtype_byte = GetTypeByte(TypeIdToType(outputs[i]->dtype_id()));
    size_t update_size = LongToSize(
      std::accumulate(output_shapes_[i].begin(), output_shapes_[i].end(), dtype_byte, std::multiplies<int64_t>()));
    outputs[i]->set_size(update_size);
  }
}

MS_ACLNN_KERNEL_FACTORY_REG(InnerUnique, InnerUniqueAscend);
}  // namespace inner_unique
}  // namespace kernel
}  // namespace mindspore
