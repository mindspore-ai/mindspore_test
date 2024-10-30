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
#include <memory>
#include "plugin/device/ascend/kernel/internal/paged_attention_mask.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
namespace mindspore {
namespace kernel {
internal::OpParamPtr InternalPagedAttentionMask::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                               const std::vector<KernelTensor *> &outputs) {
  internal::OpParamPtr param_ptr = std::make_shared<internal::OpParam>();
  param_ptr->opId = internal::OpId::PagedAttention;
  internal::MixParam op_param;
  op_param.maskType = internal::MixParam::MaskType::MASK_TYPE_NONE;
  if (!(inputs[kIndex7]->GetType()->isa<TypeNone>())) {
    op_param.maskType = internal::MixParam::MaskType::MASK_TYPE_ALIBI;
  }
  param_ptr->specificParam = op_param;
  return param_ptr;
}
}  // namespace kernel
}  // namespace mindspore
