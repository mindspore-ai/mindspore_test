/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include <algorithm>

#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/gather_pre_rms_norm.h"

namespace mindspore {
namespace ops {

void GatherPreRmsNormFuncImpl::CheckInputs(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &x_info = input_infos[kGatherPreRmsNormInputXIndex];
  auto &res_in_info = input_infos[kGatherPreRmsNormInputResInIndex];
  auto &indices_info = input_infos[kGatherPreRmsNormInputIndicesIndex];
  auto any_dim = abstract::Shape::kShapeDimAny;

  if (x_info->IsNone() || res_in_info->IsNone() || indices_info->IsNone()) {
    MS_EXCEPTION(ShapeError) << "For op[" << op_name << "], the inputs x, res_in or indices should have real "
                             << "shape, but get None!";
    return;
  }

  CheckRank(x_info, kInputRankSize2, op_name, "x");
  CheckRank(res_in_info, kInputRankSize2, op_name, "res_in");
  CheckRank(indices_info, kInputRankSize1, op_name, "indices");
  const auto &x_shp = x_info->GetShape();
  const auto &res_in_shp = res_in_info->GetShape();
  const auto &indices_shp = indices_info->GetShape();
  bool is_dynamic_shape = (x_shp.front() == any_dim) || (indices_shp.front() == any_dim);
  if (!is_dynamic_shape && (x_shp.front() != indices_shp.front())) {
    MS_EXCEPTION(ShapeError) << "For op [" << op_name << "], the first dim of x and indices must be the same.";
  }
}

ShapeArray GatherPreRmsNormFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  // Get input tensor shape.
  auto &x_info = input_infos[kGatherPreRmsNormInputXIndex];
  const auto &x_shp = x_info->GetShape();

  if (!x_info->IsDynamicRank()) {
    CheckInputs(primitive, input_infos);
  }

  return {x_shp, x_shp};
}

TypeIdList GatherPreRmsNormFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  const std::set<TypeId> x_valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16};
  const std::set<TypeId> indices_valid_types = {kNumberTypeInt32};
  auto x_type = input_infos[kGatherPreRmsNormInputXIndex]->GetType();
  auto res_in_type = input_infos[kGatherPreRmsNormInputResInIndex]->GetType();
  auto indices_type = input_infos[kGatherPreRmsNormInputIndicesIndex]->GetType();
  auto gamma_type = input_infos[kGatherPreRmsNormInputGammaIndex]->GetType();

  CheckType(x_valid_types, x_type, op_name, "x");
  CheckType(x_valid_types, res_in_type, op_name, "res_in");
  CheckType(indices_valid_types, indices_type, op_name, "indices");
  CheckType(x_valid_types, gamma_type, op_name, "gamma");

  return {kNumberTypeFloat32, x_type};
}
}  // namespace ops
}  // namespace mindspore
