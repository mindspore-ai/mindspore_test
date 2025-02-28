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

#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/adaptive_avg_pool1d.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kInputNumDims2 = 2;
}
ShapeArray AdaptiveAvgPool1DFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &x_tensor = input_infos[kInputIndex0];
  auto x_shape = x_tensor->GetShape();

  if (!x_tensor->IsDynamicRank()) {
    const int64_t input_num_dims = SizeToLong(x_shape.size());
    CheckAndConvertUtils::CheckInRange("dim of x", input_num_dims, kIncludeBoth, {2, 3}, op_name);
  } else {
    return {x_shape};
  }

  if (!x_tensor->IsDynamic()) {
    for (size_t i = 0; i < x_shape.size(); i++) {
      (void)CheckAndConvertUtils::CheckInteger(std::to_string(i) + "th dimension of x", x_shape[i], kGreaterEqual, 1,
                                               op_name);
    }
  }

  auto &output_size_info_ptr = input_infos[kInputIndex1];
  auto output_size_opt = output_size_info_ptr->GetArrayValue<int64_t>();
  const int64_t input_num_dims = SizeToLong(x_shape.size());
  if (!output_size_opt.has_value()) {
    if (input_num_dims == kInputNumDims2) {
      ShapeVector dyn_output{x_shape[0], abstract::Shape::kShapeDimAny};
      return {dyn_output};
    } else {
      ShapeVector dyn_output{x_shape[0], x_shape[1], abstract::Shape::kShapeDimAny};
      return {dyn_output};
    }
  }
  auto output_size_array_value = output_size_opt.value();
  if (output_size_array_value.IsValueUnknown(kIndex0)) {
    return {x_shape};
  }
  if (output_size_array_value.size() != 1) {
    MS_EXCEPTION(ValueError) << "For Primitive[AdaptiveAvgPool1d], the length of 'output_size' must be 1, but got"
                             << output_size_array_value.size() << "!";
  }
  x_shape[input_num_dims - 1] = static_cast<int64_t>(output_size_array_value[kIndex0]);
  return {x_shape};
}

std::vector<TypeId> AdaptiveAvgPool1DFuncImpl::InferType(const PrimitivePtr &primitive,
                                                         const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
