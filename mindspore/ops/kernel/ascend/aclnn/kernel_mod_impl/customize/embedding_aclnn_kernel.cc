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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/embedding_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace kernel {
namespace embedding {

void EmbeddingAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  ClearOpsWorkSpaceList();
  auto max_norm_opt = inputs[kIndex3]->GetValue<float>();
  if (max_norm_opt.has_value()) {
    do_renorm_ = true;
    max_norm_ = static_cast<double>(max_norm_opt.value());
    norm_type_ = static_cast<double>(device::ascend::ConvertKernelTensor<float>(inputs[kIndex4]));
    GetWorkspaceForResizeEmbeddingRenorm(inputs[1], inputs[0], max_norm_, norm_type_);
  }
  GetWorkspaceForResizeEmbedding(inputs[kIndex1], inputs[kIndex0], outputs[kIndex0]);
}

bool EmbeddingAscend::Launch(const std::vector<KernelTensor *> &inputs, const std::vector<KernelTensor *> &workspace,
                             const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (do_renorm_) {
    RunOpEmbeddingRenorm(stream_ptr, workspace, inputs[kIndex1], inputs[kIndex0], max_norm_, norm_type_);
  }
  RunOpEmbedding(stream_ptr, workspace, inputs[kIndex1], inputs[kIndex0], outputs[kIndex0]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(Embedding, EmbeddingAscend);
}  // namespace embedding
}  // namespace kernel
}  // namespace mindspore
