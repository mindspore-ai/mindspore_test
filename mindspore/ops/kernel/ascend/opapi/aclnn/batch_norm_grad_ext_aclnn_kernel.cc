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
#include "kernel/ascend/opapi/aclnn/batch_norm_grad_ext_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/core/include/mindapi/base/types.h"

namespace mindspore {
namespace kernel {
namespace batch_norm_grad_ext {

void BatchNormGradExtAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                              const std::vector<KernelTensor *> &outputs) {
  training_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex7]);
  eps_ = static_cast<double>(device::ascend::ConvertKernelTensor<pyfloat>(inputs[kIndex8]));
  output_mask_.clear();
  const auto &output_mask_vec = device::ascend::ConvertKernelTensor<std::vector<int64_t>>(inputs[kIndex9]);
  (void)std::transform(output_mask_vec.begin(), output_mask_vec.end(), std::back_inserter(output_mask_),
                       [](const int64_t &value) { return static_cast<uint8_t>(value); });

  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
                        inputs[kIndex5], inputs[kIndex6], training_, eps_, output_mask_, outputs[kIndex0],
                        outputs[kIndex1], outputs[kIndex2]);
}

bool BatchNormGradExtAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                    const std::vector<KernelTensor *> &workspace,
                                    const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], inputs[kIndex4],
        inputs[kIndex5], inputs[kIndex6], training_, eps_, output_mask_, outputs[kIndex0], outputs[kIndex1],
        outputs[kIndex2]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(BatchNormGradExt, BatchNormGradExtAscend);
}  // namespace batch_norm_grad_ext
}  // namespace kernel
}  // namespace mindspore
