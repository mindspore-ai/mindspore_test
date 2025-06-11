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

#include "infer/ops_func_impl/moe_token_permute.h"
#include <string>
#include <set>
#include <algorithm>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kDefaultNumTopk = 1;
constexpr int64_t kTokensDim = 2;
constexpr int64_t kIndices1D = 1;
constexpr int64_t kIndices2D = 2;
}  // namespace

ShapeArray MoeTokenPermuteFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto &tokens_tensor = input_infos[kIndex0];
  auto &indices_tensor = input_infos[kIndex1];
  auto tokens_shape = tokens_tensor->GetShape();
  auto indices_shape = indices_tensor->GetShape();

  auto padded_mode = input_infos[kIndex3]->GetScalarValue<bool>();
  if (padded_mode.has_value()) {
    if (padded_mode.value()) {
      MS_EXCEPTION(ValueError) << "For primitive[MoeTokenPermute], it only support padded_mode is false.";
    }
  }

  ShapeVector output_permuted_tokens_shape = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
  ShapeVector output_sorted_indices_shape = {abstract::Shape::kShapeDimAny};

  auto num_out_tokens_none = input_infos[kIndex2]->IsNone();
  int64_t num_out_tokens = 0;
  if (!num_out_tokens_none) {
    auto num_out_tokens_opt = input_infos[kIndex2]->GetScalarValue<int64_t>();
    if (!num_out_tokens_opt.has_value()) {
      return {output_permuted_tokens_shape, output_sorted_indices_shape};
    } else {
      num_out_tokens = num_out_tokens_opt.value();
      (void)CheckAndConvertUtils::CheckInteger("num_out_tokens", num_out_tokens, kGreaterEqual, 0, "MoeTokenPermute");
    }
  }
  if (!IsDynamic(tokens_shape)) {
    auto token_rank = SizeToLong(tokens_shape.size());
    if (token_rank != kTokensDim) {
      MS_EXCEPTION(ValueError) << "For primitive[MoeTokenPermute], input 'tokens' should "
                               << "be 2 dimensional, but got " << token_rank << "-dimensional.";
    }
    output_permuted_tokens_shape[kIndex1] = tokens_shape[kIndex1];
  }
  if (!IsDynamic(indices_shape)) {
    auto indices_rank = SizeToLong(indices_shape.size());
    if (indices_rank != kIndices1D && indices_rank != kIndices2D) {
      MS_EXCEPTION(ValueError) << "For primitive[MoeTokenPermute], input 'indices' should "
                               << "be 1-D or 2-D, but got " << indices_rank << "-dimensional.";
    }
    auto topk = kDefaultNumTopk;
    if (indices_rank == kIndices2D) {
      CheckAndConvertUtils::CheckInRange("dim 1 of indices", indices_shape[kIndex1], kIncludeBoth, {1, 512},
                                         "MoeTokenPermute");
      topk = indices_shape[kIndex1];
    }
    auto total_length = topk * indices_shape[kIndex0];
    num_out_tokens = (num_out_tokens <= 0) ? total_length : num_out_tokens;
    num_out_tokens = std::min(num_out_tokens, total_length);
    output_permuted_tokens_shape[kIndex0] = num_out_tokens;
    output_sorted_indices_shape[kIndex0] = total_length;
  }
  return {output_permuted_tokens_shape, output_sorted_indices_shape};
}

std::vector<TypeId> MoeTokenPermuteFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  TypeId tokens_type = input_infos[kIndex0]->GetType();
  TypeId indices_type = input_infos[kIndex1]->GetType();
  if (indices_type != kNumberTypeInt32 && indices_type != kNumberTypeInt64) {
    MS_EXCEPTION(TypeError) << "For primitive[MoeTokenPermute], indices dtype is invalid"
                            << " , should be int32 or int64.";
  }

  return {tokens_type, kNumberTypeInt32};
}
}  // namespace ops
}  // namespace mindspore
