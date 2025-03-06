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
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/adaptive_avg_pool3d_ext.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kOutputSizeLen = 3;
constexpr int64_t kPyValueNone = -1;
}  // namespace
ShapeArray AdaptiveAvgPool3DExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &x_tensor = input_infos[kInputIndex0];
  auto x_shape = x_tensor->GetShape();
  if (x_tensor->IsDynamicRank()) {
    return {x_shape};
  }
  const int64_t input_num_dims = SizeToLong(x_shape.size());
  CheckAndConvertUtils::CheckInRange("the rank of x", input_num_dims, kIncludeBoth, {kDim4, kDim5}, op_name);
  if (!x_tensor->IsDynamic()) {
    for (size_t i = 0; i < x_shape.size(); i++) {
      (void)CheckAndConvertUtils::CheckInteger(std::to_string(i) + "th dimension of x", x_shape[i], kGreaterEqual, 1,
                                               op_name);
    }
  }

  auto &output_size_info_ptr = input_infos[kInputIndex1];
  auto output_size_opt = output_size_info_ptr->GetArrayValue<int64_t>();
  (void)CheckAndConvertUtils::CheckInteger("length of output_size", SizeToLong(output_size_opt.value().size()), kEqual,
                                           kOutputSizeLen, op_name);
  auto output_size_array_value = output_size_opt.value();

  if (!output_size_opt.has_value()) {
    if (input_num_dims == kDim4) {
      ShapeVector dyn_output{x_shape[kInputIndex0], abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                             abstract::Shape::kShapeDimAny};
      return {dyn_output};
    } else {
      ShapeVector dyn_output{x_shape[kInputIndex0], x_shape[kInputIndex1], abstract::Shape::kShapeDimAny,
                             abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
      return {dyn_output};
    }
  }
  ShapeVector output_size = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                             abstract::Shape::kShapeDimAny};
  if (!output_size_array_value.IsValueUnknown(kInputIndex0) && !output_size_array_value.IsValueUnknown(kInputIndex1)) {
    output_size = output_size_array_value.ToVector();
  }

  // Update the output shape by output size and input shape.
  auto input_size_iter = x_shape.rbegin();
  auto output_size_iter = output_size.rbegin();
  for (; output_size_iter != output_size.rend(); ++output_size_iter, ++input_size_iter) {
    // If output size is none, the input shape should be used.
    if (*output_size_iter != kPyValueNone) {
      *input_size_iter = *output_size_iter;
    }
  }
  return {x_shape};
}
std::vector<TypeId> AdaptiveAvgPool3DExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                            const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kInputIndex0]->GetType();
  auto prim_name = primitive->name();
  const std::set<TypeId> valid_types = {kNumberTypeFloat16,  kNumberTypeFloat32,   kNumberTypeFloat64,
                                        kNumberTypeBFloat16, kNumberTypeComplex64, kNumberTypeComplex128};
  (void)CheckAndConvertUtils::CheckTypeIdValid("input", input_type, valid_types, prim_name);
  return {input_type};
}
}  // namespace ops
}  // namespace mindspore
