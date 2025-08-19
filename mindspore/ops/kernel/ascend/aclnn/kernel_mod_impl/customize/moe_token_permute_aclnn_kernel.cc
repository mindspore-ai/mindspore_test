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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/moe_token_permute_aclnn_kernel.h"
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
namespace moe_token_permute {
void MoeTokenPermuteAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                             const std::vector<KernelTensor *> &outputs) {
  padded_mode = inputs[kIndex3]->GetValueWithCheck<bool>();
  auto num_out_tokens_type = inputs[kIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(num_out_tokens_type);
  auto num_out_tokens_type_id = num_out_tokens_type->type_id();
  if (num_out_tokens_type_id != kMetaTypeNone) {
    num_out_tokens = device::ascend::ConvertKernelTensor<int64_t>(inputs[kIndex2]);
  }
  GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], num_out_tokens, padded_mode, outputs[kIndex0],
                        outputs[kIndex1]);
}

bool MoeTokenPermuteAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                   const std::vector<KernelTensor *> &workspace,
                                   const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], num_out_tokens, padded_mode, outputs[kIndex0],
        outputs[kIndex1]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MoeTokenPermute, MoeTokenPermuteAscend);
}  // namespace moe_token_permute
}  // namespace kernel
}  // namespace mindspore
