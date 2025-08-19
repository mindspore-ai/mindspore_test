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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/linalg_vector_norm_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace linalg_vector_norm {
void LinalgVectorNormAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  auto ord_dtype_id = inputs[kIndex1]->dtype_id();
  auto ord_value = (ord_dtype_id == kNumberTypeFloat32)
                     ? inputs[kIndex1]->GetValueWithCheck<float>()
                     : static_cast<float>(inputs[kIndex1]->GetValueWithCheck<double>());
  MAKE_SCALAR(ord_value, kNumberTypeFloat32, ord_scalar_);
  const auto dim_opt = inputs[kIndex2]->GetOptionalValueWithCheck<std::vector<int64_t>>();
  if (dim_opt.has_value()) {
    dim_ = dim_opt.value();
  }
  keepdim_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex3]);
  dtype_ = outputs[kIndex0]->dtype_id();

  GetWorkspaceForResize(inputs[kIndex0], ord_scalar_, dim_, keepdim_, dtype_, outputs[kIndex0]);
}

bool LinalgVectorNormAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], ord_scalar_, dim_, keepdim_, dtype_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(LinalgVectorNorm, LinalgVectorNormAscend);
}  // namespace linalg_vector_norm
}  // namespace kernel
}  // namespace mindspore
