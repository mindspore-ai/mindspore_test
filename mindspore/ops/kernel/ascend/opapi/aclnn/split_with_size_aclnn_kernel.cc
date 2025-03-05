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
#include "kernel/ascend/opapi/aclnn/split_with_size_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

int64_t SplitWithSizeAscend::GetDimValue(KernelTensor *axis_ptr) const noexcept {
  auto axis_vec = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(axis_ptr);
  auto dim = axis_vec[0];
  return dim;
}

bool SplitWithSizeAscend::IsTuple(const KernelTensor *tensor) {
  if (tensor == nullptr) {
    return false;
  }
  bool is_tuple = tensor->type_id() == kObjectTypeTuple;
  return is_tuple;
}

std::vector<KernelTensor *> SplitWithSizeAscend::GetSplitRealOutputs(const std::vector<KernelTensor *> &outputs) {
  if (outputs.empty()) {
    MS_LOG(EXCEPTION) << "The outputs of 'Split' should not be empty.";
  }
  std::vector<KernelTensor *> split_results;
  for (auto &output : outputs) {
    if (IsTuple(output)) {
      converted_output_ = device::ascend::ConvertKernelTensor<std::vector<KernelTensorPtr>>(output);
      std::transform(converted_output_.begin(), converted_output_.end(), std::back_inserter(split_results),
                     [](const KernelTensorPtr &tensor) -> KernelTensor * { return tensor.get(); });
    } else {
      split_results.push_back(output);
    }
  }
  return split_results;
}

void SplitWithSizeAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &outputs) {
  axis_ = GetDimValue(inputs[kIndex2]);
  dims_ = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex1]);
  std::vector<KernelTensor *> split_outputs = GetSplitRealOutputs(outputs);
  GetWorkspaceForResize(inputs[kIndex0], dims_, axis_, split_outputs);
}

bool SplitWithSizeAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                 const std::vector<KernelTensor *> &workspace,
                                 const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  std::vector<KernelTensor *> split_outputs = GetSplitRealOutputs(outputs);
  RunOp(stream_ptr, workspace, inputs[kIndex0], dims_, axis_, split_outputs);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SplitWithSize, SplitWithSizeAscend);
}  // namespace kernel
}  // namespace mindspore
