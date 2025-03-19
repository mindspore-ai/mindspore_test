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

#include "infer/ops_func_impl/roi_align_grad_ext.h"
#include <vector>
#include <string>
#include <set>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray RoiAlignGradExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &dout = input_infos[kInputIndex0];
  auto dout_shape = dout->GetShape();
  auto &boxes = input_infos[kInputIndex1];
  auto boxes_shape = boxes->GetShape();

  int64_t out_c;
  if (dout->IsDynamicRank()) {
    out_c = abstract::Shape::kShapeDimAny;
  } else {
    constexpr int64_t kGradDoutShapeSize = 4;
    CheckAndConvertUtils::CheckInteger("rank of dout shape", SizeToLong(dout_shape.size()), kEqual, kGradDoutShapeSize,
                                       op_name);
    out_c = dout_shape[kIndex1];
  }
  if (!boxes->IsDynamicRank()) {
    constexpr int64_t kBoxesShapeSize = 2;
    CheckAndConvertUtils::CheckInteger("rank of boxes shape", SizeToLong(boxes_shape.size()), kEqual, kBoxesShapeSize,
                                       op_name);
    auto boxes_second_dim = boxes_shape[kIndex1];
    if (boxes_second_dim != abstract::Shape::kShapeDimAny) {
      constexpr int64_t kBoxesShapeSecondDim = 5;
      CheckAndConvertUtils::CheckInteger("second dim of boxes shape", boxes_second_dim, kEqual, kBoxesShapeSecondDim,
                                         op_name);
    }
  }

  int64_t out_b = abstract::Shape::kShapeDimAny;
  int64_t input_height = abstract::Shape::kShapeDimAny;
  int64_t input_width = abstract::Shape::kShapeDimAny;
  auto &input_shape_info = input_infos[kInputIndex2];
  auto input_shape_opt = input_shape_info->GetArrayValue<int64_t>();
  if (input_shape_opt.has_value()) {
    auto input_shape_array = input_shape_opt.value();
    if (!input_shape_array.HasUnknownValue()) {
      auto input_shape_vector = input_shape_array.ToVector();
      constexpr int64_t kInputShapeSize = 4;
      CheckAndConvertUtils::CheckInteger("rank of inputShape", SizeToLong(input_shape_vector.size()), kEqual,
                                         kInputShapeSize, op_name);
      out_b = input_shape_vector[kIndex0];
      input_height = input_shape_vector[kIndex2];
      input_width = input_shape_vector[kIndex3];
    }
  }

  ShapeVector output_shape{out_b, out_c, input_height, input_width};
  return {output_shape};
}

std::vector<TypeId> RoiAlignGradExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto dout_type = input_infos[kInputIndex0]->GetType();
  auto boxes_type = input_infos[kInputIndex1]->GetType();
  const std::set<TypeId> valid_types = {kNumberTypeFloat32};
  CheckAndConvertUtils::CheckTypeIdValid("dout", dout_type, valid_types, op_name);
  CheckAndConvertUtils::CheckTypeIdValid("boxes", boxes_type, valid_types, op_name);
  return {dout_type};
}
}  // namespace ops
}  // namespace mindspore
