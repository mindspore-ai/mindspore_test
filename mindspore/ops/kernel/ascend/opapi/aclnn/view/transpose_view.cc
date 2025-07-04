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
#include "kernel/ascend/opapi/aclnn/view/transpose_view.h"

#include "kernel/ascend/opapi/aclnn/view/view_utils.h"
#include "mindspore/ops/view/transpose_strides_calc.h"
#include "mindspore/ops/view/view_strides_calculator.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void TransposeView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);
  const auto &dims = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  auto shape = inputs[kIndex0]->GetShapeVector();
  auto size = shape.size();
  if (dims.size() != size) {
    MS_LOG(EXCEPTION) << "DIMS should be same with shape size which is " << dims << " ,and shape " << shape;
  }
  info_ = ops::TransposeStridesCalc(old_info, dims);
  outputs[kIndex0]->set_tensor_storage_info(info_[0]);
}

void TransposeView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

bool TransposeView::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(TransposeView, TransposeView);
}  // namespace kernel
}  // namespace mindspore
