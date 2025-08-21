/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/inplace_bernoulli_scalar_aclnn_kernel.h"
#include <memory>
#include <vector>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace inplace_bernoulli_scalar {

void InplaceBernoulliScalarAclnnKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                            const std::vector<KernelTensor *> &outputs) {
  p_scalar_ = device::ascend::ConvertKernelTensor<ScalarPtr>(inputs[kIndex1]);
  seed_value_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  offset_value_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  GetWorkspaceForResize(inputs[kIndex0], p_scalar_, seed_value_, offset_value_);
}

bool InplaceBernoulliScalarAclnnKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                                  const std::vector<KernelTensor *> &workspace,
                                                  const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], p_scalar_, seed_value_, offset_value_);
  return true;
}
MS_ACLNN_KERNEL_FACTORY_REG(InplaceBernoulliScalar, InplaceBernoulliScalarAclnnKernelMod);
}  // namespace inplace_bernoulli_scalar
}  // namespace kernel
}  // namespace mindspore
