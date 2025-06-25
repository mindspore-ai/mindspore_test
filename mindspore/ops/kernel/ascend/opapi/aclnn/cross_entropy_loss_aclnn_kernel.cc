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
#include "kernel/ascend/opapi/aclnn/cross_entropy_loss_aclnn_kernel.h"
#include <tuple>
#include <algorithm>
#include <vector>
#include <map>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/acl_helper.h"
#include "kernel/ascend/acl_ir/acl_convert.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace cross_entropy_loss {
void CrossEntropyLossAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  const auto &reduction_imm = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex3]);
  reduction_ = ops::ConvertReductionStrForAclnn(reduction_imm);
  ignore_index_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex4]);
  label_smoothing_ = static_cast<double>(device::ascend::ConvertKernelTensor<double>(inputs[kIndex5]));
  lse_square_scale_for_zloss_ = static_cast<double>(device::ascend::ConvertKernelTensor<double>(inputs[kIndex6]));
  return_zloss_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex7]);
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], reduction_, ignore_index_, label_smoothing_,
                        lse_square_scale_for_zloss_, return_zloss_, outputs[kIndex0], outputs[kIndex1],
                        outputs[kIndex2], outputs[kIndex3]);
}

bool CrossEntropyLossAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], reduction_, ignore_index_,
        label_smoothing_, lse_square_scale_for_zloss_, return_zloss_, outputs[kIndex0], outputs[kIndex1],
        outputs[kIndex2], outputs[kIndex3]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(CrossEntropyLoss, CrossEntropyLossAscend);
}  // namespace cross_entropy_loss
}  // namespace kernel
}  // namespace mindspore
