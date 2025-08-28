/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include <string>
#include <memory>
#include "infer/ops_func_impl/rotated_iou.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kRotatedIouInputNums = 7;
constexpr int64_t kRotatedIouInputDims = 3;
constexpr int64_t shapeFirstDim = 0;
constexpr int64_t shapeSecondDim = 1;
constexpr int64_t shapeThirdDim = 2;
constexpr int64_t shapeThirdDimSize = 5;

void IsShapeValid(const std::string &prim_name, const ShapeVector &shape) {
  if (shape[shapeSecondDim] != abstract::Shape::kShapeDimAny) {
    (void)CheckAndConvertUtils::CheckInteger("shape third Dim", shape[shapeSecondDim], kEqual, shapeThirdDimSize,
                                             prim_name);
  }

  if (shape.size() != kRotatedIouInputDims) {
    MS_EXCEPTION(ValueError) << "For RotatedIou, the input shape dimension size must be 3. But got shape size = "
                             << shape.size() << ".";
  }
}
}  // namespace
BaseShapePtr RotatedIouFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto boxes_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto query_boxes_shape_ptr = input_args[kInputIndex1]->GetShape();

  auto boxes_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(boxes_shape_ptr);
  auto query_boxes_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(query_boxes_shape_ptr);

  auto boxes_shape = boxes_shape_map[kShape];
  auto query_boxes_shape = query_boxes_shape_map[kShape];
  if (IsDynamicRank(boxes_shape) && IsDynamicRank(query_boxes_shape)) {
    ShapeVector ret_shape = {-1, -1, -1};
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  if (IsDynamicRank(boxes_shape)) {
    IsShapeValid(prim_name, query_boxes_shape);
    ShapeVector ret_shape = {query_boxes_shape[shapeFirstDim], -1, query_boxes_shape[shapeThirdDim]};
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  if (IsDynamicRank(query_boxes_shape)) {
    IsShapeValid(prim_name, boxes_shape);
    ShapeVector ret_shape = {boxes_shape[shapeFirstDim], boxes_shape[shapeThirdDim], -1};
    return std::make_shared<abstract::Shape>(ret_shape);
  }

  IsShapeValid(prim_name, boxes_shape);
  IsShapeValid(prim_name, query_boxes_shape);
  if ((boxes_shape[shapeFirstDim] != abstract::Shape::kShapeDimAny &&
       query_boxes_shape[shapeFirstDim] != abstract::Shape::kShapeDimAny) &&
      boxes_shape[shapeFirstDim] != query_boxes_shape[shapeFirstDim]) {
    MS_EXCEPTION(ValueError) << "For RotatedIou, the input boxes_shape, query_boxes_shape must have the same "
                             << "first dim size. But boxes first dim size = " << boxes_shape[shapeFirstDim]
                             << ", query_boxes first dim size = " << query_boxes_shape[shapeFirstDim] << ".";
  }

  auto shape_first_dim = (boxes_shape[shapeFirstDim] != abstract::Shape::kShapeDimAny)
                           ? boxes_shape[shapeFirstDim]
                           : query_boxes_shape[shapeFirstDim];
  ShapeVector ret_shape = {shape_first_dim, boxes_shape[shapeThirdDim], query_boxes_shape[shapeThirdDim]};
  return std::make_shared<abstract::Shape>(ret_shape);
}

TypePtr RotatedIouFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kInputIndex0]->GetType()->Clone();
}
}  // namespace ops
}  // namespace mindspore
