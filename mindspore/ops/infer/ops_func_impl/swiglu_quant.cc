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
#include <vector>
#include <algorithm>

#include "infer/ops_func_impl/swiglu_quant.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr int64_t kMaxDim = 8192;
}  // namespace
void SwigluQuantFuncImpl::CheckShapes(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &x_info = input_infos[kIndex0];

  const auto &x_shp = x_info->GetShape();
  if (x_shp.size() <= kSize1) {
    MS_EXCEPTION(ShapeError) << "For op [" << op_name << "], the rank of input x must > 1, but got " << x_shp.size();
  }
  if (x_shp.back() != abstract::Shape::kShapeDimAny) {
    if (x_shp.back() % kSize2 != 0) {
      MS_EXCEPTION(ShapeError) << "For op [" << op_name << "], the last dim of input x must be divisible by 2, but got "
                               << x_shp.back();
    }
    if (x_shp.back() > kMaxDim) {
      MS_EXCEPTION(ShapeError) << "For op [" << op_name << "], the last dim of input x must <= " << kMaxDim
                               << "but got " << x_shp.back();
    }
  }
}

ShapeArray SwigluQuantFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &x_info = input_infos[kIndex0];

  auto any_dim = abstract::Shape::kShapeDimAny;
  auto any_shape = abstract::TensorShape::kShapeRankAny;
  if (x_info->IsDynamicRank()) {
    return {{any_shape}, {any_shape}};
  }

  CheckShapes(primitive, input_infos);

  const auto &x_shp = x_info->GetShape();
  ShapeVector scale_shp(x_shp.begin(), x_shp.end() - kSize1);
  bool last_dim_any = x_shp.back() == any_dim;
  if (last_dim_any) {
    return {x_shp, scale_shp};
  }

  ShapeVector y_shp = x_shp;
  y_shp.back() /= kSize2;
  return {y_shp, scale_shp};
}

TypeIdList SwigluQuantFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  const std::set<TypeId> support_x_types = {kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32};
  const std::set<TypeId> support_smooth_scale_types = {kNumberTypeFloat32};
  const std::set<TypeId> support_group_index_types = {kNumberTypeInt32};

  auto x_type = input_infos[kIndex0]->GetType();
  CheckType(support_x_types, x_type, op_name, "x");

  if (!input_infos[kIndex1]->IsNone()) {
    CheckType(support_smooth_scale_types, input_infos[kIndex1]->GetType(), op_name, "smooth_scale");
  }
  if (!input_infos[kIndex2]->IsNone()) {
    CheckType(support_smooth_scale_types, input_infos[kIndex2]->GetType(), op_name, "offset");
  }
  if (!input_infos[kIndex3]->IsNone()) {
    CheckType(support_group_index_types, input_infos[kIndex3]->GetType(), op_name, "group_index");
  }

  TypeId y_out_type = kNumberTypeInt8;
  TypeId scale_out_type = kNumberTypeFloat32;

  return {y_out_type, scale_out_type};
}
}  // namespace ops
}  // namespace mindspore
