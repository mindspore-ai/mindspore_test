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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/soft_margin_loss_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace kernel {
namespace soft_margin_loss {

void SoftMarginLossAscendKernelMod::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                     const std::vector<KernelTensor *> &outputs) {
  const auto &reduction_imm = static_cast<Reduction>(device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]));
  // transform reduction enum value to corresponding value
  reduction_value_ = ops::ConvertReductionForAclnn(reduction_imm);

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], reduction_value_, outputs[kIndex0]);
}

bool SoftMarginLossAscendKernelMod::Launch(const std::vector<KernelTensor *> &inputs,
                                           const std::vector<KernelTensor *> &workspace,
                                           const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], reduction_value_, outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(SoftMarginLoss, SoftMarginLossAscendKernelMod);
}  // namespace soft_margin_loss
}  // namespace kernel
}  // namespace mindspore
