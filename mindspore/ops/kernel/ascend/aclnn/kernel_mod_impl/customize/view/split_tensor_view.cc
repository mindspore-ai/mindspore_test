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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/view/split_tensor_view.h"

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/view/view_utils.h"
#include "mindspore/ops/view/split_tensor_strides_calc.h"

namespace mindspore {
namespace kernel {

void SplitTensorView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);

  auto split_size = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  auto dim = inputs[kIndex2]->GetValueWithCheck<int64_t>();

  info_ = ops::SplitTensorStridesCalc(old_info->old_shape, old_info->old_strides,
                                      inputs[kIndex0]->tensor_storage_info(), split_size, dim);
  for (size_t i = 0; i < outputs.size(); i++) {
    outputs[i]->set_tensor_storage_info(info_[i]);
  }
}

void SplitTensorView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

bool SplitTensorView::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SplitTensorView, SplitTensorView);
}  // namespace kernel
}  // namespace mindspore
