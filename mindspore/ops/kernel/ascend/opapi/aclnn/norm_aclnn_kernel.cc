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
#include "kernel/ascend/opapi/aclnn/norm_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr float kDefaultOrd = 2.0;
}
void NormAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                  const std::vector<KernelTensor *> &outputs) {
  auto p_dtype_id = inputs[kIndex1]->dtype_id();
  if (p_dtype_id == kMetaTypeNone) {
    MAKE_SCALAR(kDefaultOrd, kNumberTypeFloat32, p_scalar_);
  } else {
    auto p_value = (p_dtype_id == kNumberTypeFloat32)
                     ? inputs[kIndex1]->GetValueWithCheck<float>()
                     : static_cast<float>(inputs[kIndex1]->GetValueWithCheck<double>());
    MAKE_SCALAR(p_value, kNumberTypeFloat32, p_scalar_);
  }

  const auto dim_opt = inputs[kIndex2]->GetOptionalValueWithCheck<std::vector<int64_t>>();
  if (dim_opt.has_value()) {
    dim_ = dim_opt.value();
  } else {
    dim_ = std::vector<int64_t>{};
  }
  keepdim_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex3]);

  GetWorkspaceForResize(inputs[kIndex0], p_scalar_, dim_, keepdim_, outputs[kIndex0]);
}

bool NormAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                        const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], p_scalar_, dim_, keepdim_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Norm, NormAscend);
}  // namespace kernel
}  // namespace mindspore
