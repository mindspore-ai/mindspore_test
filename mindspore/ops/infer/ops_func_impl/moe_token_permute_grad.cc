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

#include "infer/ops_func_impl/moe_token_permute_grad.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kPermutedDim = 2;
constexpr int64_t kSortedIndices1D = 1;
}  // namespace

ShapeArray MoeTokenPermuteGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  auto &permuted_output_grad_tensor = input_infos[kIndex0];
  auto &sorted_indices_tensor = input_infos[kIndex1];
  auto permuted_output_grad_shape = permuted_output_grad_tensor->GetShape();
  auto sorted_indices_shape = sorted_indices_tensor->GetShape();
  auto padded_mode = input_infos[kIndex3]->GetScalarValue<bool>();
  if (padded_mode.has_value()) {
    if (padded_mode.value()) {
      MS_EXCEPTION(ValueError) << "For primitive[MoeTokenPermuteGrad], it only support 'padded_mode' is false.";
    }
  }

  ShapeVector output_permuted_tokens_shape = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
  if (!IsDynamic(permuted_output_grad_shape)) {
    auto permuted_output_grad_rank = SizeToLong(permuted_output_grad_shape.size());
    if (permuted_output_grad_rank != kPermutedDim) {
      MS_EXCEPTION(ValueError) << "For primitive[MoeTokenPermuteGrad], input 'permuted_output_grad' should "
                               << "be 2 dimensional, but got " << permuted_output_grad_rank << "-dimensional.";
    }
    output_permuted_tokens_shape[kIndex1] = permuted_output_grad_shape[kIndex1];
  }
  auto num_topk = input_infos[kIndex2]->GetScalarValue<int64_t>();
  if (!num_topk.has_value()) {
    return {output_permuted_tokens_shape};
  }
  if (!IsDynamic(sorted_indices_shape)) {
    auto sorted_indices_rank = SizeToLong(sorted_indices_shape.size());
    if (sorted_indices_rank != kSortedIndices1D) {
      MS_EXCEPTION(ValueError) << "For primitive[MoeTokenPermuteGrad], input 'sorted_indices' should "
                               << "be 1-D Tensor, but got " << sorted_indices_rank << "-dimensional.";
    }
    auto topk = num_topk.value();
    if (topk <= 0) {
      MS_EXCEPTION(ValueError) << "For primitive[MoeTokenPermuteGrad], 'num_tok' should "
                               << "be great than 0, but got: " << topk;
    }
    auto tokens_num = sorted_indices_shape[0] / topk;
    output_permuted_tokens_shape[0] = tokens_num;
  }
  return {output_permuted_tokens_shape};
}

std::vector<TypeId> MoeTokenPermuteGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                           const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
