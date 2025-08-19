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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/inner_inplace_index_put_aclnn_kernel.h"
#include <algorithm>
#include <utility>
#include <functional>
#include "ir/tensor.h"
#include "abstract/ops/primitive_infer_map.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace inner_inplace_index_put {
namespace {
constexpr size_t kInnerInplaceIndexPutKenelMinNum = 4;
constexpr size_t kInnerInplaceIndexPutEmptyShape = 9;
}  // namespace

void InnerInplaceIndexPutAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &outputs) {
  if (MS_UNLIKELY(inputs.size() < kInnerInplaceIndexPutKenelMinNum)) {
    MS_LOG(EXCEPTION) << "For 'InnerInplaceIndexPut', inputs should be " << kInnerInplaceIndexPutKenelMinNum
                      << " at least, bug got " << inputs.size();
  }
  auto last_kernel_tensor = *(inputs.end() - kIndex1);
  MS_EXCEPTION_IF_NULL(last_kernel_tensor);
  accumulate_ = device::ascend::ConvertKernelTensor<bool>(last_kernel_tensor);
  auto value_tensor = *(inputs.end() - kIndex2);
  MS_EXCEPTION_IF_NULL(value_tensor);
  std::vector<KernelTensor *> indices(inputs.begin() + kIndex1, inputs.end() - kIndex2);
  indices = RemoveTrailingEmptyTensor(indices);
  GetWorkspaceForResize(inputs[kIndex0], indices, value_tensor, accumulate_, outputs[kIndex0]);
}

bool InnerInplaceIndexPutAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &workspace,
                                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto input_tensor = inputs[kIndex0];
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto input_shape = input_tensor->GetShape()->GetShapeVector();
  auto value_tensor = *(inputs.end() - kIndex2);
  MS_EXCEPTION_IF_NULL(value_tensor);
  auto value_shape = value_tensor->GetShape()->GetShapeVector();
  std::vector<KernelTensor *> indices(inputs.begin() + kIndex1, inputs.end() - kIndex2);
  indices = RemoveTrailingEmptyTensor(indices);
  // If it's empty tensor doesn't deal with it.
  auto input_numel = std::accumulate(input_shape.begin(), input_shape.end(), kIndex1, std::multiplies<int64_t>());
  auto values_numel = std::accumulate(value_shape.begin(), value_shape.end(), kIndex1, std::multiplies<int64_t>());
  auto indices_nums = indices.size();
  if (input_numel == 0 || values_numel == 0 || indices_nums == 0) {
    return true;
  }

  RunOp(stream_ptr, workspace, inputs[kIndex0], indices, value_tensor, accumulate_, outputs[kIndex0]);
  return true;
}

// Remove empty tensors from the end of the indices list.
std::vector<KernelTensor *> InnerInplaceIndexPutAscend::RemoveTrailingEmptyTensor(
  const std::vector<KernelTensor *> &indices) {
  std::vector<KernelTensor *> new_indices = indices;
  while (!new_indices.empty()) {
    auto back_shape = new_indices.back()->GetShape()->GetShapeVector();
    if (back_shape.size() == kInnerInplaceIndexPutEmptyShape &&
        std::all_of(back_shape.begin(), back_shape.end(), [](int i) { return i == 0; })) {
      new_indices.pop_back();
    } else {
      break;
    }
  }
  return new_indices;
}

MS_ACLNN_KERNEL_FACTORY_REG(InnerInplaceIndexPut, InnerInplaceIndexPutAscend);
}  // namespace inner_inplace_index_put
}  // namespace kernel
}  // namespace mindspore
