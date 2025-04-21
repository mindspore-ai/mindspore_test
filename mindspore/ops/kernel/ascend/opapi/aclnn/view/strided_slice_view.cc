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
#include "kernel/ascend/opapi/aclnn/view/strided_slice_view.h"

#include "kernel/ascend/opapi/aclnn/view/view_utils.h"
#include "mindspore/ops/view/stridedslice_strides_calc.h"
#include "mindspore/ops/view/view_strides_calculator.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void StridedSliceView::UpdateTensorInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);
  auto begin = inputs[kIndex1]->GetValueWithCheck<std::vector<int64_t>>();
  auto end = inputs[kIndex2]->GetValueWithCheck<std::vector<int64_t>>();
  auto step = inputs[kIndex3]->GetValueWithCheck<std::vector<int64_t>>();
  if (IsDynamic(step) || begin.size() != end.size() || begin.size() != step.size() || ops::HasZero(step)) {
    MS_LOG(EXCEPTION) << "StridedSliceView is not supported. Please check the input parameters";
    return;
  }
  auto shape = inputs[kIndex0]->GetShapeVector();
  auto size = shape.size();
  info_ = ops::StridedSliceStridesCalc(old_info, size, &shape, &begin, &end, &step);
  outputs[kIndex0]->set_tensor_storage_info(info_[0]);
}

void StridedSliceView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  UpdateTensorInfo(inputs, outputs);
}

bool StridedSliceView::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(InnerStridedSliceView, StridedSliceView);
}  // namespace kernel
}  // namespace mindspore
