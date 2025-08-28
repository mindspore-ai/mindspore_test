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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/view/view.h"

#include "kernel/ascend/aclnn/kernel_mod_impl/customize/view/view_utils.h"
#include "mindspore/ops/view/view_strides_calc.h"
#include "mindspore/ops/view/view_strides_calculator.h"

namespace mindspore {
namespace kernel {

void ViewView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                      const std::vector<KernelTensor *> &outputs) {
  auto ori_storage_info = inputs[0]->tensor_storage_info();
  if (ori_storage_info != nullptr && !ori_storage_info->is_contiguous) {
    MS_LOG(EXCEPTION) << "input tensor is not contiguous, storage info:" << ori_storage_info->ToString();
  }

  const auto &shape = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  if (std::any_of(shape.begin(), shape.end(), [](const int &shape_i) { return shape_i < -1; })) {
    MS_EXCEPTION(ValueError) << "For View the component of shape can't be less than -1, but got " << shape;
  }
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);

  auto infos =
    ops::ViewStridesCalc(old_info->old_shape, old_info->old_strides, inputs[kIndex0]->tensor_storage_info(), shape);
  infos[0]->ori_size = inputs[0]->size();
  outputs[kIndex0]->set_tensor_storage_info(infos[0]);
  GEN_EXECUTOR_FOR_VIEW(op_type_, inputs, outputs);
}

void ViewView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

bool ViewView::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                      const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(ViewView, ViewView);
}  // namespace kernel
}  // namespace mindspore
