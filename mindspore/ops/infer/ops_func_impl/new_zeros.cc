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

#include <set>
#include "infer/ops_func_impl/new_zeros.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
ShapeArray NewZerosFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto shape_opt = input_infos[kIndex1]->GetArrayValue<int64_t>();
  if (!shape_opt.has_value()) {
    return {{abstract::Shape::kShapeRankAny}};
  }

  const auto shape = shape_opt.value();
  ShapeVector output_shape;
  for (size_t i = 0; i < shape.size(); i++) {
    if (shape.IsValueUnknown(i)) {
      output_shape.push_back(abstract::TensorShape::kShapeDimAny);
    } else {
      int64_t shape_i = shape[i];
      MS_CHECK_VALUE(shape_i >= 0,
                     CheckAndConvertUtils::FormatCheckIntegerMsg(std::to_string(i) + "th dimension of input shape",
                                                                 shape_i, kGreaterEqual, 0, primitive));
      output_shape.push_back(shape_i);
    }
  }

  return {output_shape};
}

std::vector<TypeId> NewZerosFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  const auto &dtype_info = input_infos[kIndex2];
  // return dtype if dtype is not none
  if (!dtype_info->IsNone()) {
    const auto dtype_opt = dtype_info->GetScalarValue<int64_t>();
    if (dtype_opt.has_value()) {
      const auto dtype = static_cast<TypeId>(dtype_opt.value());
      return {dtype};
    }
  }
  // or else return input tensor's dtype
  const auto input_tensor_type = input_infos[kIndex0]->GetType();
  return {input_tensor_type};
}
}  // namespace mindspore::ops
