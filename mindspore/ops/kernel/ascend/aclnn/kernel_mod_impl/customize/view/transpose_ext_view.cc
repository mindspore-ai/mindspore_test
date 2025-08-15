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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/view/transpose_ext_view.h"

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/view/view_utils.h"
#include "mindspore/ops/view/transpose_ext_view_strides_calc.h"
#include "mindspore/ops/view/view_strides_calculator.h"

namespace mindspore {
namespace kernel {
void TransposeExtView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);
  const auto &dim0 = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  const auto &dim1 = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  auto infos = ops::TransposeExtViewStridesCalc(old_info->old_shape, old_info->old_strides,
                                                inputs[kIndex0]->tensor_storage_info(), dim0, dim1);
  outputs[kIndex0]->set_tensor_storage_info(infos[0]);
}

void TransposeExtView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

bool TransposeExtView::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(TransposeExtView, TransposeExtView);
}  // namespace kernel
}  // namespace mindspore
