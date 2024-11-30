/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "infer/ops_func_impl/std.h"
#include <set>
#include <memory>
#include <vector>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/reduce_arithmetic.h"

namespace mindspore {
namespace ops {
ShapeArray StdFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &input = input_infos[kInputIndex0];
  const auto &input_shape = input->GetShape();
  const auto input_shape_size = input_shape.size();
  const auto &dim = input_infos[kInputIndex1];
  const auto &keepdim_opt = input_infos[kInputIndex3]->GetScalarValue<bool>();

  if (!keepdim_opt.has_value()) {
    return ShapeArray{ShapeVector({abstract::TensorShape::kShapeRankAny})};
  }
  const auto keepdim = keepdim_opt.value();
  auto is_dim_none_or_empty = dim->IsNone();
  if (!is_dim_none_or_empty) {
    const auto &dim_opt = dim->GetArrayValue<int64_t>();
    if (!dim_opt.has_value()) {
      return ShapeArray{ShapeVector({abstract::TensorShape::kShapeRankAny})};
    }
    is_dim_none_or_empty = dim_opt.value().size() == 0;
  }

  if (input->IsDynamicRank()) {
    return is_dim_none_or_empty && !keepdim ? ShapeArray{ShapeVector({})}
                                            : ShapeArray{ShapeVector({abstract::TensorShape::kShapeRankAny})};
  }
  if (is_dim_none_or_empty) {
    return keepdim ? ShapeArray{ShapeVector(input_shape_size, 1)} : ShapeArray{ShapeVector({})};
  }

  const auto &dim_opt = dim->GetArrayValue<int64_t>();
  if (dim_opt.value().HasUnknownValue()) {
    return ShapeArray{ShapeVector({abstract::TensorShape::kShapeRankAny})};
  }
  const auto &dim_vector = dim_opt.value().ToVector();
  std::vector<int64_t> real_dim_vector;
  (void)std::transform(
    dim_vector.begin(), dim_vector.end(), std::back_inserter(real_dim_vector),
    [&input_shape_size, &primitive](const int64_t &dim) { return CalRealAixs(dim, input_shape_size, primitive); });
  const auto &out_shape = ReduceFuncCalShapeInferImpl(primitive, input_shape, real_dim_vector, keepdim);
  return {out_shape};
}

std::vector<TypeId> StdFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};
  const auto &input_type = input_infos[kInputIndex0]->GetType();
  const auto &prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTypeIdValid("input", input_type, valid_types, prim_name);
  return {input_type};
}

}  // namespace ops
}  // namespace mindspore
