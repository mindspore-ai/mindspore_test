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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/inner_index_aclnn_kernel.h"
#include <algorithm>
#include <utility>
#include <functional>
#include <memory>
#include "ir/tensor.h"
#include "abstract/ops/primitive_infer_map.h"
#include "kernel/ascend/acl_ir/acl_convert.h"

namespace mindspore {
namespace kernel {
namespace inner_index {
namespace {
constexpr size_t kInnerIndexMinNum = 2;
constexpr size_t kInnerIndexEmptyShape = 9;

}  // namespace

std::vector<KernelTensor *> InnerIndexAscend::GetInnerIndexRealInputs(const std::vector<KernelTensor *> &inputs) {
  if (MS_UNLIKELY(inputs.size() < kInnerIndexMinNum)) {
    MS_LOG(EXCEPTION) << "For 'InnerIndex', inputs should be " << kInnerIndexMinNum << " at least, bug got "
                      << inputs.size();
  }
  std::vector<KernelTensor *> tensors(inputs.begin() + kIndex1, inputs.end());
  std::vector<KernelTensor *> new_tensors;
  for (auto &tensor : tensors) {
    auto shape = tensor->GetShape()->GetShapeVector();
    if (shape.size() == kInnerIndexEmptyShape &&
        std::all_of(shape.begin(), shape.end(), [](int i) { return i == 0; })) {
      auto tensor_shape = std::make_shared<abstract::TensorShape>();
      tensor_shape->SetShapeVector({0});
      KernelTensor *empty_shape = new KernelTensor();
      empty_shape->SetType(tensor->GetType());
      empty_shape->SetShape(tensor_shape);
      new_tensors.push_back(empty_shape);
    } else {
      new_tensors.push_back(tensor);
    }
  }
  return new_tensors;
}
void InnerIndexAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                        const std::vector<KernelTensor *> &outputs) {
  auto indices = GetInnerIndexRealInputs(inputs);
  GetWorkspaceForResize(inputs[kIndex0], indices, outputs[kIndex0]);
}

bool InnerIndexAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                              const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  auto indices = GetInnerIndexRealInputs(inputs);
  RunOp(stream_ptr, workspace, inputs[kIndex0], indices, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(InnerIndex, InnerIndexAscend);
}  // namespace inner_index
}  // namespace kernel
}  // namespace mindspore
