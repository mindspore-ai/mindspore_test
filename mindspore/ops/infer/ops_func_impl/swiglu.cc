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

#include "infer/ops_func_impl/swiglu.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray SwigluFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  std::optional<int64_t> dim_opt = input_infos[kInputIndex1]->GetScalarValue<int64_t>();
  constexpr size_t kSplitNum = 2;
  auto &x = input_infos[kInputIndex0];
  ShapeVector x_shape = x->GetShape();
  if (MS_LIKELY(x->IsDynamicRank())) {
    return {x_shape};
  }
  auto rank = SizeToLong(x_shape.size());
  if (MS_UNLIKELY(!dim_opt.has_value())) {
    ShapeVector dyn_output = ShapeVector(rank, abstract::TensorShape::kShapeDimAny);
    return {dyn_output};
  } else {
    auto dim_temp = dim_opt.value();
    MS_CHECK_VALUE(dim_temp >= -rank && dim_temp < rank, CheckAndConvertUtils::FormatCheckInRangeMsg(
                                                           "dim", dim_temp, kIncludeLeft, {-rank, rank}, primitive));
    if (dim_temp < 0) {
      dim_temp += rank;
    }
    if (x_shape[dim_temp] != abstract::TensorShape::kShapeDimAny) {
      if (!(x_shape[dim_temp] % kSplitNum)) {
        x_shape[dim_temp] = x_shape[dim_temp] / kSplitNum;
      } else {
        MS_EXCEPTION(ValueError)
          << "For 'Swiglu', the dimension specified by 'dim' must be divisible by 2, but got x_shape[dim]: "
          << x_shape[dim_temp] << " .";
      }
    }
    return {x_shape};
  }
}

std::vector<TypeId> SwigluFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  auto type = input_infos[kInputIndex0]->GetType();
  return {type};
}
}  // namespace ops
}  // namespace mindspore
