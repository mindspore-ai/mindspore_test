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

#include "infer/ops_func_impl/roi_align_ext.h"
#include <vector>
#include <string>
#include <set>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray RoiAlignExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &input = input_infos[kInputIndex0];
  auto input_shape = input->GetShape();
  auto &boxes = input_infos[kInputIndex1];
  auto boxes_shape = boxes->GetShape();

  int64_t out_c;
  int64_t out_k;
  if (input->IsDynamicRank()) {
    out_c = abstract::Shape::kShapeDimAny;
  } else {
    constexpr int64_t kInputShapeSize = 4;
    CheckAndConvertUtils::CheckInteger("rank of input shape", SizeToLong(input_shape.size()), kEqual, kInputShapeSize,
                                       op_name);
    out_c = input_shape[kIndex1];
  }
  if (boxes->IsDynamicRank()) {
    out_k = abstract::Shape::kShapeDimAny;
  } else {
    constexpr int64_t kBoxesShapeSize = 2;
    CheckAndConvertUtils::CheckInteger("rank of boxes shape", SizeToLong(boxes_shape.size()), kEqual, kBoxesShapeSize,
                                       op_name);
    auto boxes_second_dim = boxes_shape[kIndex1];
    if (boxes_second_dim != abstract::Shape::kShapeDimAny) {
      constexpr int64_t kBoxesShapeSecondDim = 5;
      CheckAndConvertUtils::CheckInteger("second dim of boxes shape", boxes_second_dim, kEqual, kBoxesShapeSecondDim,
                                         op_name);
    }
    out_k = boxes_shape[kIndex0];
  }

  int64_t pooled_height = abstract::Shape::kShapeDimAny;
  int64_t pooled_width = abstract::Shape::kShapeDimAny;
  auto &output_size_info = input_infos[kInputIndex2];
  auto output_size_opt = output_size_info->GetArrayValue<int64_t>();
  if (output_size_opt.has_value()) {
    auto output_size_array = output_size_opt.value();
    if (!output_size_array.HasUnknownValue()) {
      auto output_size_vector = output_size_array.ToVector();
      CheckAndConvertUtils::CheckInRange("rank of output_size", SizeToLong(output_size_vector.size()), kIncludeBoth,
                                         {kDim1, kDim2}, op_name);
      pooled_height = output_size_vector[kIndex0];
      pooled_width = output_size_vector.size() == kDim2 ? output_size_vector[kIndex1] : pooled_height;
    }
  }

  ShapeVector output_shape{out_k, out_c, pooled_height, pooled_width};
  return {output_shape};
}

std::vector<TypeId> RoiAlignExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto input_type = input_infos[kInputIndex0]->GetType();
  auto boxes_type = input_infos[kInputIndex1]->GetType();
  const std::set<TypeId> valid_types = {kNumberTypeFloat32};
  CheckAndConvertUtils::CheckTypeIdValid("input", input_type, valid_types, op_name);
  CheckAndConvertUtils::CheckTypeIdValid("boxes", boxes_type, valid_types, op_name);
  return {input_type};
}
}  // namespace ops
}  // namespace mindspore
