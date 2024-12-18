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
#include "kernel/ascend/opapi/aclnn/view/transpose_ext_view.h"

#include <functional>

#include "kernel/ascend/opapi/aclnn/view/view_utils.h"
#include "mindspore/ops/view/transpose_ext_strides_calc.h"
#include "mindspore/ops/view/view_strides_calculator.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
namespace {
size_t GetOriginInputSize(const ops::OldTensorInfoPtr old_info, TypeId type_id) {
  const auto &ori_shape = old_info->ori_shape;
  auto num = std::accumulate(ori_shape.begin(), ori_shape.end(), int64_t(1), std::multiplies<int64_t>());
  auto ori_size = abstract::TypeIdSize(type_id) * LongToSize(num);
  return ori_size;
}
}  // namespace
void TransposeExtView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[kIndex0]);
  const auto &dim0 = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  const auto &dim1 = inputs[kIndex2]->GetValueWithCheck<int64_t>();
  info_ = ops::TransposeExtStridesCalc(old_info, dim0, dim1);
  info_[0]->ori_size = GetOriginInputSize(inputs[0]);
  outputs[kIndex0]->set_tensor_storage_info(info_[0]);
  GEN_EXECUTOR_FOR_VIEW(op_type_, inputs, outputs);
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
