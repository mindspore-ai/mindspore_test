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

#include "infer/ops_func_impl/rms_norm.h"

#include <string>
#include <map>
#include "abstract/dshape.h"
#include "ops/op_def.h"
#include "op_def/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace ops {
BaseShapePtr RmsNormFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_ptr = input_args[0]->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();
  auto x_rank = x_shape.size();
  auto gamma_shape_ptr = input_args[1]->GetShape();
  auto gamma_shape = gamma_shape_ptr->GetShapeVector();
  auto gamma_rank = gamma_shape.size();
  MS_EXCEPTION_IF_CHECK_FAIL(!IsShapeNone(x_shape) && !IsShapeNone(gamma_shape),
                             "For RmsNorm, [gamma] or [input] is none tensor, which is not allowed.");
  auto rstd_shape = x_shape;
  auto out_shape = x_shape;
  if (IsDynamicRank(gamma_shape)) {
    if (!IsDynamicRank(x_shape)) {
      rstd_shape = ShapeVector(x_rank, abstract::TensorShape::kShapeDimAny);
    } else {
      rstd_shape = ShapeVector{abstract::TensorShape::kShapeRankAny};
    }
  } else if (!IsDynamicRank(x_shape)) {
    MS_EXCEPTION_IF_CHECK_FAIL(gamma_rank <= x_rank,
                               "For RmsNorm, The [gamma] rank can not be bigger than the [input] rank."
                               "But got: " +
                                 std::to_string(gamma_rank) + " vs " + std::to_string(x_rank));
    for (auto dim = x_rank - gamma_rank; dim < x_rank; dim++) {
      int64_t x_dim = x_shape[dim];
      int64_t gamma_dim = gamma_shape[dim - x_rank + gamma_rank];
      MS_EXCEPTION_IF_CHECK_FAIL(
        x_dim == gamma_dim || x_dim == abstract::TensorShape::kShapeDimAny ||
          gamma_dim == abstract::TensorShape::kShapeDimAny,
        "For RmsNorm, Each dimension of [gamma] must be aligned to the corresponding dimension of [input]. "
        "But got: " +
          std::to_string(gamma_dim) + " vs " + std::to_string(x_dim));
      rstd_shape[dim] = 1;
      if (x_dim == abstract::TensorShape::kShapeDimAny && gamma_dim != abstract::TensorShape::kShapeDimAny) {
        out_shape[dim] = gamma_dim;
      }
    }
  }
  auto y_output_ptr = std::make_shared<abstract::Shape>(out_shape);
  auto rstd_output_ptr = std::make_shared<abstract::Shape>(rstd_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{y_output_ptr, rstd_output_ptr});
}

TypePtr RmsNormFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto x_dtype = input_args[kInputIndex0]->GetType();
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_dtype, std::make_shared<TensorType>(kFloat32)});
}

int32_t RmsNormFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto epsilon_value = GetScalarValue<pyfloat>(input_args[kInputIndex2]->GetValue());
  if (MS_UNLIKELY(!epsilon_value.has_value())) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(epsilon_value.value() > 0 && epsilon_value.value() <= 1,
                 CheckAndConvertUtils::FormatCheckInRangeMsg<pyfloat>("epsilon", epsilon_value.value(), kIncludeRight,
                                                                      {0., 1.}, primitive));
  return OP_CHECK_SUCCESS;
}

ShapeArray RmsNormFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto x_shape = x_tensor->shape();
  auto x_rank = x_shape.size();
  const auto &gamma_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(gamma_tensor);
  const auto gamma_shape = gamma_tensor->shape();
  auto gamma_rank = gamma_shape.size();
  MS_EXCEPTION_IF_CHECK_FAIL(!IsShapeNone(x_shape) && !IsShapeNone(gamma_shape),
                             "For RmsNorm, [gamma] or [input] is none tensor, which is not allowed.");
  auto rstd_shape = x_shape;
  MS_EXCEPTION_IF_CHECK_FAIL(gamma_rank <= x_rank,
                             "For RmsNorm, The [gamma] rank can not be bigger than the [input] rank."
                             "But got: " +
                               std::to_string(gamma_rank) + " vs " + std::to_string(x_rank));
  for (auto dim = x_rank - gamma_rank; dim < x_rank; dim++) {
    MS_EXCEPTION_IF_CHECK_FAIL(
      x_shape[dim] == gamma_shape[dim - x_rank + gamma_rank],
      "For RmsNorm, Each dimension of [gamma] must be aligned to the corresponding dimension of [input]. "
      "But got: " +
        std::to_string(x_shape[dim]) + " vs " + std::to_string(gamma_shape[dim - x_rank + gamma_rank]));
    rstd_shape[dim] = 1;
  }
  return {x_shape, rstd_shape};
}

TypePtrList RmsNormFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_dtype = x_tensor->Dtype();
  return {x_dtype, kFloat32};
}

REGISTER_SIMPLE_INFER(kNameRmsNorm, RmsNormFuncImpl);
}  // namespace ops
}  // namespace mindspore
