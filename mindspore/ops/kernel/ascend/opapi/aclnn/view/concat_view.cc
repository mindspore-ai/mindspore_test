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
#include "kernel/ascend/opapi/aclnn/view/concat_view.h"
#include <memory>
#include <functional>
#include "kernel/ascend/opapi/aclnn/view/view_utils.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {

void ConcatView::UpdateOutputTensorInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  auto input_num = inputs.size();
  size_t offset = 0;
  auto input_type = inputs[0]->dtype_id();
  auto type_size = GetTypeByte(TypeIdToType(input_type));
  auto ori_shape = outputs[0]->GetShapeVector();
  auto ori_size = std::accumulate(ori_shape.begin(), ori_shape.end(), 1, std::multiplies<int64_t>()) * type_size;
  for (size_t i = 0; i < input_num - 1; ++i) {
    ops::OldTensorInfoPtr old_info = GetOldTensorInfo(inputs[i]);
    auto new_storage_info = std::make_shared<TensorStorageInfo>(
      inputs[i]->GetShapeVector(), old_info->ori_strides, offset, ori_shape, old_info->ori_strides, true, ori_size);
    inputs[i]->set_tensor_storage_info(new_storage_info);
    offset += inputs[i]->size() / type_size;
  }
  GEN_EXECUTOR_FOR_VIEW(op_type_, inputs, outputs);
}

void ConcatView::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  UpdateOutputTensorInfo(inputs, outputs);
}

bool ConcatView::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(ConcatView, ConcatView);
}  // namespace kernel
}  // namespace mindspore
