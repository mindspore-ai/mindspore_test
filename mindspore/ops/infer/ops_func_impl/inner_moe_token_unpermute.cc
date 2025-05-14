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

#include "infer/ops_func_impl/inner_moe_token_unpermute.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kDefaultTopk = 1;
constexpr int64_t kUnpermutedDim = 2;
constexpr int64_t kUnpermutedIndiceDim = 1;
}  // namespace

ShapeArray InnerMoeTokenUnpermuteFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  auto &permuted_tokens_tensor = input_infos[kIndex0];
  auto &sorted_indices_tensor = input_infos[kIndex1];
  auto permuted_tokens_shape = permuted_tokens_tensor->GetShape();
  auto sorted_indices_shape = sorted_indices_tensor->GetShape();

  auto padded_mode = input_infos[kIndex3]->GetScalarValue<bool>();
  if (padded_mode.has_value()) {
    if (padded_mode.value()) {
      MS_EXCEPTION(ValueError) << "For primitive[MoeTokenUnpermute], it only support padded_mode is false.";
    }
  }

  ShapeVector output_shape = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
  if (!IsDynamic(permuted_tokens_shape)) {
    auto permuted_token_rank = SizeToLong(permuted_tokens_shape.size());
    if (permuted_token_rank != kUnpermutedDim) {
      MS_EXCEPTION(ValueError) << "For primitive[MoeTokenUnpermute], the dims of input permuted_tokens should "
                               << "be 2 dimensional, but got " << permuted_token_rank << "-dimensional.";
    }
    output_shape[kIndex1] = permuted_tokens_shape[permuted_token_rank - 1];
  }
  if (!IsDynamic(sorted_indices_shape)) {
    auto sorted_indices_rank = SizeToLong(sorted_indices_shape.size());
    if (sorted_indices_rank != kUnpermutedIndiceDim) {
      MS_EXCEPTION(ValueError) << "For primitive[MoeTokenUnpermute], the dims of input sorted_indices should "
                               << "be 1 dimensional, but got " << sorted_indices_rank << "-dimensional.";
    }
    auto &probs_tensor = input_infos[kIndex2];
    if (!probs_tensor->IsNone()) {
      auto probs_shape = probs_tensor->GetShape();
      if (!IsDynamic(probs_shape)) {
        auto probs_rank = SizeToLong(probs_shape.size());
        if (probs_rank != kUnpermutedDim) {
          MS_EXCEPTION(ValueError) << "For primitive[MoeTokenUnpermute], the dims of input probs should "
                                   << "be 2 dimensional, but got " << sorted_indices_rank << "-dimensional.";
        }
        auto topk = probs_shape[1];
        if (topk <= 0) {
          MS_EXCEPTION(ValueError) << "For primitive[MoeTokenUnpermute], the shape of input probs should "
                                   << "be great than 0, but got " << topk;
        }
        auto num_unpermuted_tokens = sorted_indices_shape[0] / topk;
        output_shape[0] = num_unpermuted_tokens;
      }
    } else {
      auto topk = kDefaultTopk;
      auto num_unpermuted_tokens = sorted_indices_shape[0] / topk;
      output_shape[0] = num_unpermuted_tokens;
    }
  }
  return {output_shape};
}

std::vector<TypeId> InnerMoeTokenUnpermuteFuncImpl::InferType(const PrimitivePtr &primitive,
                                                              const InferInfoPtrList &input_infos) const {
  TypeId unpermute_token_type = input_infos[kIndex0]->GetType();
  TypeId sorted_indices_type = input_infos[kIndex1]->GetType();
  if (sorted_indices_type != kNumberTypeInt32) {
    MS_EXCEPTION(TypeError) << "For primitive[MoeTokenUnpermute], sorted_indices dtype is invalid"
                            << " , should be int32.";
  }
  return {unpermute_token_type};
}
}  // namespace ops
}  // namespace mindspore
