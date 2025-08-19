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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/view/diagonal_view.h"

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/view/view_utils.h"
#include "mindspore/ops/view/diagonal_strides_calc.h"

namespace mindspore {
namespace kernel {

void DiagonalView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);
  auto offset = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  auto dim1 = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  auto dim2 = inputs[kIndex3]->GetValueWithCheck<int64_t>();
  info_ = ops::DiagonalStridesCalc(old_info, offset, dim1, dim2);
  outputs[kIndex0]->set_tensor_storage_info(info_[0]);
}

void DiagonalView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

bool DiagonalView::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(DiagonalView, DiagonalView);
}  // namespace kernel
}  // namespace mindspore
