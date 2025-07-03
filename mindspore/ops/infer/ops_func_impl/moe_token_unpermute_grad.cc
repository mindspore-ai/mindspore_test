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

#include "infer/ops_func_impl/moe_token_unpermute_grad.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {

ShapeArray MoeTokenUnpermuteGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  auto &permuted_tokens_tensor = input_infos[kIndex0];
  auto &probs_tensor = input_infos[kIndex3];
  auto permuted_tokens_shape = permuted_tokens_tensor->GetShape();

  ShapeVector probs_shape = {};
  if (!probs_tensor->IsNone()) {
    probs_shape = probs_tensor->GetShape();
  }

  return {permuted_tokens_shape, probs_shape};
}

std::vector<TypeId> MoeTokenUnpermuteGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                             const InferInfoPtrList &input_infos) const {
  auto permuted_token_type = input_infos[kIndex0]->GetType();
  auto unpermuted_token_type = input_infos[kIndex1]->GetType();
  if (permuted_token_type != unpermuted_token_type) {
    MS_EXCEPTION(TypeError) << "For primitive [MoeTokenUnpermuteGrad], the permuted_tokens and unpermuted_tokens_grad"
                            << " must have the same data type. But got permuted_tokens type is " << permuted_token_type
                            << ", unpermuted_tokens_grad type is " << unpermuted_token_type;
  }
  return {permuted_token_type, permuted_token_type};
}
}  // namespace ops
}  // namespace mindspore
