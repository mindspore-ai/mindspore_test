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
#include "infer/ops_func_impl/matrix_determinant.h"

#include <memory>

#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
BaseShapePtr MatrixDeterminantFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  const auto x_shape = input_args[kIndex0]->GetShape();
  const auto x_shape_vec = x_shape->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec))) {
    return std::make_shared<abstract::TensorShape>(ShapeVector{abstract::TensorShape::kShapeRankAny});
  }
  constexpr size_t kMinRank = 2;
  constexpr int64_t kMinDim = 2;
  auto x_rank = x_shape_vec.size();
  MS_CHECK_VALUE(x_rank >= kMinRank,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("x rank", x_rank, kGreaterEqual, kMinRank, primitive));
  auto x_row = x_shape_vec[x_rank - kIndex1];
  auto x_col = x_shape_vec[x_rank - kIndex2];
  auto output_shape =
    std::make_shared<abstract::TensorShape>(ShapeVector(x_shape_vec.begin(), x_shape_vec.end() - kIndex2));
  if (MS_UNLIKELY(x_row == abstract::TensorShape::kShapeDimAny || x_col == abstract::TensorShape::kShapeDimAny)) {
    return output_shape;
  }
  MS_CHECK_VALUE(x_row == x_col,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("row size", x_row, kEqual, x_col, primitive));
  MS_CHECK_VALUE(x_row >= kMinDim,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("row size", x_row, kGreaterEqual, kMinDim, primitive));
  MS_CHECK_VALUE(x_col >= kMinDim,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("column size", x_col, kGreaterEqual, kMinDim, primitive));
  return output_shape;
}

TypePtr MatrixDeterminantFuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}
}  // namespace ops
}  // namespace mindspore
