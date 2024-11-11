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

#include "infer/ops_func_impl/std_mean.h"
#include <vector>
#include <string>
#include <set>
#include "ops_utils/op_utils.h"
#include "infer/ops_func_impl/reduce_arithmetic.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray StdMeanFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  MS_LOG(DEBUG) << "Run ReduceExtandGeneralInferShape_" << primitive->name() << " start";
  const auto &input = input_infos[kInputIndex0];
  const auto input_shape = input->GetShape();
  const auto input_shape_size = input_shape.size();

  const auto keepdim_opt = input_infos[kInputIndex3]->GetScalarValue<bool>();
  if (MS_UNLIKELY(!keepdim_opt.has_value())) {
    return ShapeArray{ShapeVector({abstract::Shape::kShapeRankAny}), ShapeVector({abstract::Shape::kShapeRankAny})};
  }
  const auto keepdim = keepdim_opt.value();

  const auto &dim = input_infos[kInputIndex1];
  // If dim is None
  if (dim->IsNone()) {
    return keepdim ? ShapeArray{ShapeVector(input_shape_size, 1), ShapeVector(input_shape_size, 1)}
                   : ShapeArray{ShapeVector({}), ShapeVector({})};
  }

  const auto &dim_opt = dim->GetArrayValue<int64_t>();
  const auto dim_size = dim_opt->size();
  if (dim_opt.has_value()) {
    // If dim is empty tuple and keepdim is False, return a zero-dimensional Tensor
    if (dim_size == 0 && !keepdim) {
      return ShapeArray{ShapeVector({}), ShapeVector({})};
    }
  }

  if (input->IsDynamicRank()) {
    return {input_shape, input_shape};
  }
  if (!dim_opt.has_value()) {
    // If dim is dynamic.
    return keepdim
             ? ShapeArray{ShapeVector(input_shape_size, -1), ShapeVector(input_shape_size, -1)}
             : ShapeArray{ShapeVector({abstract::Shape::kShapeRankAny}), ShapeVector({abstract::Shape::kShapeRankAny})};
  }

  const auto dim_array = dim_opt.value();
  // All values of the dim are known.
  if (!dim_array.HasUnknownValue()) {
    std::vector<int64_t> dim_vector = dim_array.ToVector();
    std::vector<int64_t> real_dim_vector;
    (void)std::transform(
      dim_vector.begin(), dim_vector.end(), std::back_inserter(real_dim_vector),
      [&input_shape_size, &primitive](const int64_t &dim) { return CalRealAixs(dim, input_shape_size, primitive); });
    auto out_shape = ReduceFuncCalShapeInferImpl(primitive, input_shape, real_dim_vector, keepdim);
    return {out_shape, out_shape};
  }

  // If the dim has unknown value, the reduction position will be any of the input dimensions.
  if (!keepdim) {
    MS_CHECK_VALUE(input_shape_size >= dim_size,
                   CheckAndConvertUtils::FormatCheckInRangeMsg("dim size", dim_size, kIncludeLeft,
                                                               {0, input_shape_size}, primitive));
    return ShapeArray{ShapeVector(input_shape_size - dim_size, -1), ShapeVector(input_shape_size - dim_size, -1)};
  }
  auto out_shape = ShapeVector(input_shape_size, -1);
  for (size_t i = 0; i < dim_array.size(); ++i) {
    if (!dim_array.IsValueUnknown(i)) {
      auto dim_i = CalRealAixs(dim_array[i], input_shape_size, primitive);
      out_shape[dim_i] = 1;
    }
  }
  MS_LOG(DEBUG) << "Run ReduceExtandGeneralInferShape_" << primitive->name() << " end";
  return {out_shape, out_shape};
}

std::vector<TypeId> StdMeanFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  const std::set<TypeId> valid_dtype_set = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};
  const auto type = input_infos[kInputIndex0]->GetType();
  const auto &prim_name = primitive->name();
  CheckAndConvertUtils::CheckTypeIdValid("input", type, valid_dtype_set, prim_name);
  return {type, type};
}
}  // namespace ops
}  // namespace mindspore
