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

#include <vector>
#include <memory>
#include <set>
#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"
#include "infer/ops_func_impl/narrow.h"

namespace mindspore::ops {
BaseShapePtr NarrowFuncImpl::InferShape(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto input_x_shape = input_args[0]->GetShape()->GetShapeVector();
  (void)CheckAndConvertUtils::CheckInteger("rank of input", SizeToLong(input_x_shape.size()), kGreaterThan, 0,
                                           prim_name);

  if (IsDynamicRank(input_x_shape)) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }

  auto axis_value_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
  auto begin_value_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  auto length_value_opt = GetScalarValue<int64_t>(input_args[kInputIndex3]->GetValue());

  auto x_rank = SizeToLong(input_x_shape.size());
  if (!axis_value_opt.has_value()) {
    auto out_shape = input_x_shape;
    for (int dim = 0; dim < x_rank; ++dim) {
      out_shape[dim] = abstract::Shape::kShapeDimAny;
    }
    return std::make_shared<abstract::TensorShape>(out_shape);
  }

  auto axis_value = axis_value_opt.value();
  if (!begin_value_opt.has_value() || !length_value_opt.has_value()) {
    auto out_shape = input_x_shape;
    if (axis_value < 0) {
      axis_value += x_rank;
    }
    out_shape[axis_value] = abstract::Shape::kShapeDimAny;
    return std::make_shared<abstract::TensorShape>(out_shape);
  }

  MS_CHECK_VALUE(axis_value >= -x_rank && axis_value < x_rank, "dim value error. dim:" + std::to_string(axis_value) +
                                                                 ", dim should be in [" + std::to_string(-x_rank) +
                                                                 ", " + std::to_string(x_rank) + ").");
  axis_value = axis_value < 0 ? axis_value + x_rank : axis_value;

  auto x_axis_size = input_x_shape[axis_value];

  if (x_axis_size == abstract::Shape::kShapeDimAny) {
    return std::make_shared<abstract::TensorShape>(input_x_shape);
  }

  auto begin_value = begin_value_opt.value();
  MS_CHECK_VALUE(begin_value >= -x_axis_size && begin_value <= x_axis_size,
                 "For primitive [Narrow]: start value error, start: " + std::to_string(begin_value) +
                   ", start should be in [" + std::to_string(-x_axis_size) + ", " + std::to_string(x_axis_size) + "].");
  begin_value = begin_value < 0 ? begin_value + x_axis_size : begin_value;

  auto length_value = length_value_opt.value();
  auto max_length = x_axis_size - begin_value;
  MS_CHECK_VALUE(length_value >= 0 && length_value <= max_length,
                 "length value error. length: " + std::to_string(length_value) + ", length should be in [0, " +
                   std::to_string(max_length) + "].");

  auto out_shape = input_x_shape;
  out_shape[axis_value] = length_value;

  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr NarrowFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType();
  const std::set<TypePtr> valid_type = {kInt8, kInt32, kInt64, kUInt8, kFloat16, kFloat32, kBool, kBFloat16};
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, valid_type, primitive->name());

  return input_type->Clone();
}
}  // namespace mindspore::ops
