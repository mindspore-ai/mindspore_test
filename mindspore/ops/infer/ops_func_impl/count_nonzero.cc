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

#include "infer/ops_func_impl/count_nonzero.h"
#include <set>
#include <memory>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_constants.h"
#include "infer/ops_func_impl/reduce_arithmetic.h"

namespace mindspore {
namespace ops {

ShapeArray CountNonZeroFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto x_shape = input_infos[kInputIndex0]->GetShape();

  // If axis is None
  if (input_infos[kInputIndex1]->IsNone()) {
    return ShapeArray{ShapeVector({})};
  }

  auto axis_array_opt = input_infos[kInputIndex1]->GetArrayValue<int64_t>();
  if (axis_array_opt.has_value() && axis_array_opt->size() == 0) {
    // If axis is empty tuple, return a zero-dimensional Tensor
    return ShapeArray{ShapeVector({})};
  }

  if (input_infos[kInputIndex0]->IsDynamicRank()) {
    return ShapeArray{x_shape};
  }

  if (!axis_array_opt.has_value()) {
    // axis is dynamic.
    return ShapeArray{ShapeVector({abstract::Shape::kShapeRankAny})};
  }

  auto x_shape_size = x_shape.size();
  auto axis_array = axis_array_opt.value();
  // All values of the axis are known.
  if (!axis_array.HasUnknownValue()) {
    std::vector<int64_t> axis_vec = axis_array.ToVector();
    std::vector<int64_t> real_axis_vec;
    (void)std::transform(
      axis_vec.begin(), axis_vec.end(), std::back_inserter(real_axis_vec),
      [&x_shape_size, &primitive](const int64_t &axis) { return CalRealAixs(axis, x_shape_size, primitive); });
    auto out_shape = ReduceFuncCalShapeInferImpl(primitive, x_shape, real_axis_vec, false);
    return ShapeArray{out_shape};
  }

  // If the axis has unknown value, the reduction position will be any of the input dimensions.
  MS_CHECK_VALUE(x_shape.size() >= axis_array_opt->size(),
                 CheckAndConvertUtils::FormatCheckInRangeMsg("axis size", axis_array_opt->size(), kIncludeLeft,
                                                             {0, x_shape.size()}, primitive));
  return ShapeArray{ShapeVector(x_shape.size() - axis_array_opt->size(), -1)};
}

std::vector<TypeId> CountNonZeroFuncImpl::InferType(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  return {kNumberTypeInt64};
}
}  // namespace ops
}  // namespace mindspore
