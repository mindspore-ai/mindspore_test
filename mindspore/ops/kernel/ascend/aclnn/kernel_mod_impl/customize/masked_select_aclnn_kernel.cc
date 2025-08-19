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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/masked_select_aclnn_kernel.h"
#include <algorithm>
#include <functional>
#include <memory>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"

namespace mindspore {
namespace kernel {
namespace masked_select {
void MaskedSelectAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], outputs[kIndex0]);
}

bool MaskedSelectAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  const auto &all_tensor = RunOpSync(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], outputs[kIndex0]);

  // Update output shape.
  outputs_shape_.resize(1);
  outputs_shape_[kIndex0] = all_tensor.at(kIndex2);
  return true;
}

void MaskedSelectAclnnKernelMod::UpdateOutputShapeAndSize(const std::vector<KernelTensor *> &,
                                                          const std::vector<KernelTensor *> &outputs) {
  outputs[kIndex0]->SetShapeVector(outputs_shape_[kIndex0]);
  size_t type_size = UnitSizeInBytes(outputs[kIndex0]->dtype_id());
  size_t size = SizeOf(outputs_shape_[kIndex0]) * type_size;
  outputs[kIndex0]->set_size(size);
}
MS_ACLNN_KERNEL_FACTORY_REG(MaskedSelect, MaskedSelectAclnnKernelMod);
}  // namespace masked_select
}  // namespace kernel
}  // namespace mindspore
