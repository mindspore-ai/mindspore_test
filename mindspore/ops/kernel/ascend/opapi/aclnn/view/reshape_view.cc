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
#include "kernel/ascend/opapi/aclnn/view/reshape_view.h"

#include "kernel/ascend/opapi/aclnn/view/view_utils.h"
#include "mindspore/ops/view/reshape_strides_calc.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void ReshapeView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &outputs) {
  // If input is not contiguous, framework will make the input contiguous in launch.
  auto ori_storage_info = inputs[0]->tensor_storage_info();
  if (ori_storage_info && !ori_storage_info->is_contiguous) {
    MS_LOG(WARNING) << "For Reshape, the input is not contiguous. It is suggested to call Contiguous "
                       "explicitly. Input shape is "
                    << ori_storage_info->shape;
    is_input_not_contiguous_ = true;
    return;
  }
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);
  auto shape = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  if (std::any_of(shape.begin(), shape.end(), [](const int &shape_i) { return shape_i < -1; })) {
    MS_EXCEPTION(ValueError) << "ReshapeView the component of shape can't be less than -1, but got " << shape;
  }

  auto ori_info = inputs[kIndex0]->tensor_storage_info();
  if (ori_info != nullptr && !ori_info->is_contiguous) {
    info_ = ops::ReshapeUncontiguousCalcImpl(old_info, shape);
    if (info_.empty()) {
      info_ = ops::ReshapeCalcImpl(old_info, shape);
    }
  } else {
    info_ = ops::ReshapeCalcImpl(old_info, shape);
  }
  info_[0]->ori_size = inputs[0]->size();
  outputs[kIndex0]->set_tensor_storage_info(info_[0]);
  GEN_EXECUTOR_FOR_VIEW(op_type_, inputs, outputs);
}

void ReshapeView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

bool ReshapeView::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  if (is_input_not_contiguous_) {
    // If input is not contiguous, framework will make a contiguous input, then we need re-compute the
    // out tensor storage info.
    UpdateOutputTensorInfo(inputs, outputs);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(ReshapeView, ReshapeView);
}  // namespace kernel
}  // namespace mindspore
