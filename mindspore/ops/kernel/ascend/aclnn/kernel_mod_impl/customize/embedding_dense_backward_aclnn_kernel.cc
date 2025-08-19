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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/embedding_dense_backward_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace embedding_dense_backward {

void EmbeddingDenseBackwardAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                    const std::vector<KernelTensor *> &outputs) {
  auto padding_idx_opt = inputs[kIndex3]->GetValue<int64_t>();
  num_weights_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  // the type of padding_idx is uint64_t in aclnnEmbeddingDenseBackward api,
  // but the type of padding_idx is int64_t in the operator belowing aclnn api where -1 indicates None value
  // this maybe a risk.
  padding_idx_ = 0xFFFFFFFF;
  if (padding_idx_opt.has_value()) {
    padding_idx_ = static_cast<uint64_t>(padding_idx_opt.value() < 0 ? padding_idx_opt.value() + num_weights_
                                                                     : padding_idx_opt.value());
  }

  scale_grad_by_freq_ = device::ascend::ConvertKernelTensor<bool>(inputs[kIndex4]);
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], num_weights_, padding_idx_, scale_grad_by_freq_, outputs[0]);
}

bool EmbeddingDenseBackwardAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &workspace,
                                          const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], num_weights_, padding_idx_, scale_grad_by_freq_,
        outputs[0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(EmbeddingDenseBackward, EmbeddingDenseBackwardAscend);
}  // namespace embedding_dense_backward
}  // namespace kernel
}  // namespace mindspore
