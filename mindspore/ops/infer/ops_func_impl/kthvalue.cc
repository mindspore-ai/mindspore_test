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

#include "infer/ops_func_impl/kthvalue.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {

namespace {
inline ShapeArray KthvalueGetOutputShapeArray(const ShapeVector &output_shape) {
  ShapeArray shape_array{output_shape, output_shape};
  return shape_array;
}
}  // namespace

ShapeArray KthvalueFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &x = input_infos[kInputIndex0];
  const ShapeVector x_shape = x->GetShape();
  if (MS_UNLIKELY(IsDynamicRank(x_shape))) {
    return KthvalueGetOutputShapeArray(x_shape);
  }
  std::optional<int64_t> keep_dims_opt = input_infos[kInputIndex3]->GetScalarValue<bool>();
  if (MS_UNLIKELY(!keep_dims_opt.has_value())) {
    ShapeVector dynamic_rank_shape{abstract::TensorShape::kShapeRankAny};
    return KthvalueGetOutputShapeArray(dynamic_rank_shape);
  }
  if (std::any_of(x_shape.begin(), x_shape.end(), [](const auto &item) { return item == 0; })) {
    MS_EXCEPTION(ValueError) << primitive->name() << " cannot deal with empty input. Please try other inputs";
  }

  auto x_rank = SizeToLong(x_shape.size());
  auto keep_dims = keep_dims_opt.value();
  auto out_dim = keep_dims ? x_rank : x_rank - 1;
  std::optional<int64_t> dim_opt = input_infos[kInputIndex2]->GetScalarValue<int64_t>();
  if (MS_UNLIKELY(!dim_opt.has_value())) {
    if (x_rank == 0) {
      out_dim = 0;
    }
    return KthvalueGetOutputShapeArray(ShapeVector(out_dim, abstract::TensorShape::kShapeDimAny));
  }
  auto dim = dim_opt.value();
  if (MS_UNLIKELY(x_rank == 0)) {
    if (dim != -1 && dim != 0) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', with 0d input tensor, 'dim' must be one of 0 or -1, but got: " << dim << ".";
    }
    return KthvalueGetOutputShapeArray(x_shape);
  }
  MS_CHECK_VALUE(dim >= -x_rank && dim < x_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("dim", dim, kIncludeLeft, {-x_rank, x_rank}, primitive));
  dim = dim < 0 ? dim + x_rank : dim;
  if (MS_UNLIKELY(x_shape[dim] == 0)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", the pos:" << dim
                             << " of input_x's shape can not be 0, but got " << x_shape;
  }

  ShapeVector output_shape(x_shape);
  if (keep_dims) {
    output_shape[dim] = 1;
  } else {
    (void)output_shape.erase(output_shape.begin() + dim);
  }
  return KthvalueGetOutputShapeArray(output_shape);
}

std::vector<TypeId> KthvalueFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  auto type = input_infos[kInputIndex0]->GetType();
  return {type, kNumberTypeInt64};
}
}  // namespace ops
}  // namespace mindspore
