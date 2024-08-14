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
#include "kernel/ascend/opapi/aclnn/rand_ext_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "transform/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {

void RandExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                     const std::vector<KernelTensor *> &outputs) {
  constexpr double from_ = 0.0;
  constexpr double to_ = 1.0;
  seed_ = static_cast<uint64_t>(transform::ConvertKernelTensor<int64_t>(inputs[kIndex1]));
  offset_ = static_cast<uint64_t>(transform::ConvertKernelTensor<int64_t>(inputs[kIndex2]));
  GetWorkspaceForResize(outputs[kIndex0], from_, to_, seed_, offset_);
}

bool RandExtAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  constexpr double from_ = 0.0;
  constexpr double to_ = 1.0;
  RunOp(stream_ptr, workspace, outputs[kIndex0], from_, to_, seed_, offset_);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(RandExt, RandExtAscend);
}  // namespace kernel
}  // namespace mindspore
