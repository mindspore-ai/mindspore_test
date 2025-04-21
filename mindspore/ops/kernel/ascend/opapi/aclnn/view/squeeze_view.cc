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
#include "kernel/ascend/opapi/aclnn/view/squeeze_view.h"

#include "kernel/ascend/opapi/aclnn/view/view_utils.h"
#include "mindspore/ops/view/reshape_strides_calc.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void SqueezeView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);
  auto shape = outputs[0]->GetShapeVector();
  info_ = ops::ReshapeCalcImpl(old_info, shape);
  info_[0]->ori_size = GetOriginInputSize(inputs[0]);
  outputs[kIndex0]->set_tensor_storage_info(info_[0]);
  GEN_EXECUTOR_FOR_VIEW(op_type_, inputs, outputs);
}

void SqueezeView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

bool SqueezeView::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SqueezeView, SqueezeView);
}  // namespace kernel
}  // namespace mindspore
