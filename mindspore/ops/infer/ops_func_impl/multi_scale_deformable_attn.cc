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

#include "infer/ops_func_impl/multi_scale_deformable_attn.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
int GetDimOfValueLocationsWeight(const InferInfoPtr &value, const InferInfoPtr &locations, const InferInfoPtr &weight,
                                 const int dim) {
  ShapeVector value_shape = value->GetShape();
  if (!value->IsDynamicRank() && value_shape[dim] != abstract::Shape::kShapeDimAny) {
    return value_shape[dim];
  }

  ShapeVector locations_shape = locations->GetShape();
  if (!locations->IsDynamicRank() && locations_shape[dim] != abstract::Shape::kShapeDimAny) {
    return locations_shape[dim];
  }

  ShapeVector weight_shape = weight->GetShape();
  if (!weight->IsDynamicRank() && weight_shape[dim] != abstract::Shape::kShapeDimAny) {
    return weight_shape[dim];
  }

  return abstract::Shape::kShapeDimAny;
}

int GetDimOfLocationsWeight(const InferInfoPtr &locations, const InferInfoPtr &weight, const int dim) {
  ShapeVector locations_shape = locations->GetShape();
  if (!locations->IsDynamicRank() && locations_shape[dim] != abstract::Shape::kShapeDimAny) {
    return locations_shape[dim];
  }

  ShapeVector weight_shape = weight->GetShape();
  if (!weight->IsDynamicRank() && weight_shape[dim] != abstract::Shape::kShapeDimAny) {
    return weight_shape[dim];
  }

  return abstract::Shape::kShapeDimAny;
}
}  // namespace

ShapeArray MultiScaleDeformableAttnFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  int ret_dim2 = abstract::Shape::kShapeDimAny;

  int shape_dim0 = 0;
  int ret_dim0 =
    GetDimOfValueLocationsWeight(input_infos[kIndex0], input_infos[kIndex3], input_infos[kIndex4], shape_dim0);

  int shape_dim1 = 1;
  int ret_dim1 = GetDimOfLocationsWeight(input_infos[kIndex3], input_infos[kIndex4], shape_dim1);

  int shape_dim2 = 2;
  int ret = GetDimOfValueLocationsWeight(input_infos[kIndex0], input_infos[kIndex3], input_infos[kIndex4], shape_dim2);
  if (ret != abstract::Shape::kShapeDimAny) {
    ShapeVector value_shape = input_infos[kIndex0]->GetShape();
    int shape_dim3 = 3;
    if (!input_infos[kIndex0]->IsDynamicRank() && value_shape[shape_dim3] != abstract::Shape::kShapeDimAny) {
      ret_dim2 = value_shape[shape_dim3] * ret;
    }
  }

  ShapeVector ret_shape = {ret_dim0, ret_dim1, ret_dim2};
  return {ret_shape};
}

std::vector<TypeId> MultiScaleDeformableAttnFuncImpl::InferType(const PrimitivePtr &primitive,
                                                                const InferInfoPtrList &input_infos) const {
  return {input_infos[0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
