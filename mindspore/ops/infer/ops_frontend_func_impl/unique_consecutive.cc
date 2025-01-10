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
#include "ops/ops_frontend_func_impl.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
namespace {
TypePtr UniqueConsecutiveInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto y_type = input_args[kIndex0]->GetType();
  auto indices_type = std::make_shared<TensorType>(kInt64);
  auto counts_type = std::make_shared<TensorType>(kInt64);
  return std::make_shared<Tuple>(std::vector<TypePtr>{y_type->Clone(), indices_type, counts_type});
}

BaseShapePtr UniqueConsecutiveFrontendInferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) {
  auto shape_x = input_args[0]->GetShape();
  MS_EXCEPTION_IF_NULL(shape_x);
  auto x_shape_vector = shape_x->GetShapeVector();
  auto return_inverse = GetScalarValue<bool>(input_args[kIndex1]->BuildValue());
  auto return_counts = GetScalarValue<bool>(input_args[kIndex2]->BuildValue());

  // dynamic rank
  if (IsDynamicRank(x_shape_vector)) {
    abstract::BaseShapePtr out_shape_ptr =
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{out_shape_ptr, out_shape_ptr, out_shape_ptr});
  }

  // dim is None
  if (input_args[kIndex3]->GetType()->isa<TypeNone>()) {
    auto out_shape_ptr = std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny});
    auto empty_shape_ptr = std::make_shared<abstract::Shape>(ShapeVector{});

    abstract::BaseShapePtrList out_shapes{};
    out_shapes.emplace_back(out_shape_ptr);

    // when both return_inverse and return_counts are true, return 3 non-empty tensors.
    if (return_inverse.has_value() && return_inverse.value()) {
      out_shapes.emplace_back(shape_x->Clone());
    } else {
      out_shapes.emplace_back(empty_shape_ptr);
    }
    if (return_counts.has_value() && return_counts.value()) {
      out_shapes.emplace_back(out_shape_ptr);
    } else {
      out_shapes.emplace_back(empty_shape_ptr);
    }
    return std::make_shared<abstract::TupleShape>(out_shapes);
  }

  // dim is an integer.
  auto dim = GetScalarValue<int64_t>(input_args[kIndex3]->BuildValue());
  if (!dim.has_value()) {
    std::vector<int64_t> y_shape_vector(x_shape_vector.size(), abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
      std::make_shared<abstract::Shape>(y_shape_vector),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny}),
      std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeDimAny})});
  }
  auto dim_value = dim.value();
  if (dim_value < -static_cast<int64_t>(x_shape_vector.size()) ||
      dim_value >= static_cast<int64_t>(x_shape_vector.size())) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the value of 'dim' should be in ["
                             << -static_cast<int64_t>(x_shape_vector.size()) << ", " << x_shape_vector.size()
                             << "), but got " << dim_value;
  }
  if (dim_value < 0) {
    dim_value += static_cast<int64_t>(x_shape_vector.size());
  }

  // indices, when return_inverse=false, its still x_shape[dim], otherwise the shape after execute in ascend will be
  // wrong
  ShapeVector indices_shape_vector = {x_shape_vector[dim_value]};
  if (!return_inverse.has_value()) {
    indices_shape_vector = {abstract::Shape::kShapeDimAny};
  }

  // y
  x_shape_vector[dim_value] = abstract::Shape::kShapeDimAny;
  abstract::BaseShapePtr y_shape_ptr = std::make_shared<abstract::Shape>(x_shape_vector);

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{
    y_shape_ptr, std::make_shared<abstract::Shape>(indices_shape_vector),
    std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeDimAny}))});
}
}  // namespace

class OPS_API UniqueConsecutiveFrontendFuncImpl : public OpFrontendFuncImpl {
 public:
  AbstractBasePtr InferAbstract(const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto infer_type = UniqueConsecutiveInferType(primitive, input_args);
    auto infer_shape = UniqueConsecutiveFrontendInferShape(primitive, input_args);
    return abstract::MakeAbstract(infer_shape, infer_type);
  }
};

REGISTER_PRIMITIVE_FUNCTION_FRONTEND_FUNC_IMPL("UniqueConsecutive", UniqueConsecutiveFrontendFuncImpl);
}  // namespace mindspore::ops
