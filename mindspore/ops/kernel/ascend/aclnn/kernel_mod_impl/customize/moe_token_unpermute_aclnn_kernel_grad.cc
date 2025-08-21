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
#include "kernel/ascend/aclnn/kernel_mod_impl/customize/moe_token_unpermute_aclnn_kernel_grad.h"

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
namespace moe_token_unpermute {
void MoeTokenUnpermuteGradAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                   const std::vector<KernelTensor *> &outputs) {
  padded_mode = inputs[kIndex4]->GetValueWithCheck<bool>();
  auto restore_shape_opt = inputs[kIndex5]->GetValue<std::vector<int64_t>>();
  std::vector<int64_t> restore_default = {1, 1};
  restore_shape_val = restore_shape_opt.has_value() ? restore_shape_opt.value() : restore_default;

  auto in_shape = inputs[kIndex0]->GetShapeVector();
  if (in_shape[kIndex0] == 0 || in_shape[kIndex1] == 0) {
    is_empty = true;
    MS_LOG(DEBUG) << "For [MoeTokenUnpermuteGradAscend], input is empty.";
  } else {
    GetWorkspaceForResize(inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], padded_mode,
                          restore_shape_val, outputs[kIndex0], outputs[kIndex1]);
  }
}

bool MoeTokenUnpermuteGradAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                         const std::vector<KernelTensor *> &workspace,
                                         const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  if (is_empty) {
    MS_LOG(DEBUG) << "For [MoeTokenUnpermuteGradAscend], input is empty. Do not launch kernel.";
  } else {
    RunOp(stream_ptr, workspace, inputs[kIndex0], inputs[kIndex1], inputs[kIndex2], inputs[kIndex3], padded_mode,
          restore_shape_val, outputs[kIndex0], outputs[kIndex1]);
  }
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(MoeTokenUnpermuteGrad, MoeTokenUnpermuteGradAscend);
}  // namespace moe_token_unpermute
}  // namespace kernel
}  // namespace mindspore
