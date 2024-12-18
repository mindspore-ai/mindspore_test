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
#include "kernel/ascend/opapi/aclnn/view/broadcast_to_view.h"

#include "kernel/ascend/opapi/aclnn/view/view_utils.h"
#include "mindspore/ops/view/broadcast_to_strides_calc.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void BroadcastToView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);
  auto shape = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();

  info_ = ops::BroadCastToStrideCalc(old_info, shape);
  info_[0]->ori_size = GetOriginInputSize(inputs[0]);
  outputs[kIndex0]->set_tensor_storage_info(info_[0]);

  GEN_EXECUTOR_FOR_VIEW(op_type_, inputs, outputs);
}

void BroadcastToView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

bool BroadcastToView::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(BroadcastToView, BroadcastToView);
}  // namespace kernel
}  // namespace mindspore
