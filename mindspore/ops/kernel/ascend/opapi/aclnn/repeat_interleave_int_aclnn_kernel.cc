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
#include "kernel/ascend/opapi/aclnn/repeat_interleave_int_aclnn_kernel.h"
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"

namespace mindspore {
namespace kernel {
namespace repeat_interleave_int {

void RepeatInterleaveIntAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                 const std::vector<KernelTensor *> &outputs) {
  auto output_shape = outputs[kIndex0]->GetShapeVector();
  auto dim_opt = inputs[kIndex2]->GetOptionalValueWithCheck<int64_t>();
  auto output_size_opt = inputs[kIndex3]->GetOptionalValueWithCheck<int64_t>();
  repeats_ = inputs[kIndex1]->GetValueWithCheck<int64_t>();
  if (dim_opt.has_value()) {
    dim_ = dim_opt.value();
    auto rank = SizeToLong(output_shape.size());
    dim_ = (dim_ < 0) ? (dim_ + rank) : dim_;
    op_type_ = "aclnnRepeatInterleaveIntWithDim";
    output_size_ = output_shape[dim_];
    GetWorkspaceForResize(inputs[kIndex0], repeats_, dim_, output_size_, outputs[kIndex0]);
  } else {
    op_type_ = "aclnnRepeatInterleaveInt";
    output_size_ = output_shape[0];
    GetWorkspaceForResize(inputs[kIndex0], repeats_, output_size_, outputs[kIndex0]);
  }
  if (output_size_opt.has_value()) {
    int64_t output_size_imm = output_size_opt.value();
    if (output_size_imm != output_size_) {
      MS_EXCEPTION(RuntimeError) << "Allocated size does not match required size.";
    }
  }
}

bool RepeatInterleaveIntAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &workspace,
                                       const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (op_type_ == "aclnnRepeatInterleaveIntWithDim") {
    RunOp(stream_ptr, workspace, inputs[kIndex0], repeats_, dim_, output_size_, outputs[kIndex0]);
  } else {
    RunOp(stream_ptr, workspace, inputs[kIndex0], repeats_, output_size_, outputs[kIndex0]);
  }

  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(RepeatInterleaveInt, RepeatInterleaveIntAscend);
}  // namespace repeat_interleave_int
}  // namespace kernel
}  // namespace mindspore
