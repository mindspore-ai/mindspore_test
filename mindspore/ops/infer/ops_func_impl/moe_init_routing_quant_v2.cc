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
#include "infer/ops_func_impl/moe_init_routing_quant_v2.h"

#include <set>
#include <vector>
#include <algorithm>

#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/convert_utils_base.h"

namespace mindspore {
namespace ops {

void MoeInitRoutingQuantV2FuncImpl::CheckInputs(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &x_info = input_infos[kIndex0];
  auto &expert_idx_info = input_infos[kIndex1];
  auto any_dim = abstract::Shape::kShapeDimAny;

  if (x_info->IsNone() || expert_idx_info->IsNone()) {
    MS_EXCEPTION(ShapeError) << "For op[" << op_name << "], the input_x or input expert_idx should have real "
                             << " shape, but get None!";
    return;
  }

  CheckRank(x_info, kInputRankSize, op_name, "x");
  CheckRank(expert_idx_info, kInputRankSize, op_name, "expert_idx");
  const auto &x_shp = x_info->GetShape();
  const auto &expert_idx_shp = expert_idx_info->GetShape();
  bool is_dynamic_shape = (x_shp.front() == any_dim) || (expert_idx_shp.front() == any_dim);
  if (is_dynamic_shape && (x_shp.front() != expert_idx_shp.front())) {
    MS_EXCEPTION(ShapeError) << "For op [" << op_name << "], the first dim of x and expert_idx must be the same.";
  }

  auto active_num = input_infos[kIndex2]->GetScalarValue<int64_t>();
  auto expert_capacity = input_infos[kIndex3]->GetScalarValue<int64_t>();
  auto expert_num = input_infos[kIndex4]->GetScalarValue<int64_t>();

  if (active_num.has_value() && active_num.value() < 0) {
    MS_EXCEPTION(ValueError) << "For op [" << op_name << "], active_num must >= 0, but got " << active_num.value();
  }
  if (expert_capacity.has_value() && expert_capacity.value() < 0) {
    MS_EXCEPTION(ValueError) << "For op [" << op_name << "], expert_capacity must >= 0, but got "
                             << expert_capacity.value();
  }
  if (expert_num.has_value() && expert_num.value() < 0) {
    MS_EXCEPTION(ValueError) << "For op [" << op_name << "], expert_num must >= 0, but got " << expert_num.value();
  }
  // quant scenario inputs check.
  auto quant_mode = input_infos[kIndex8]->GetScalarValue<int64_t>();
  auto &quant_scale = input_infos[kIndex9];
  if (quant_mode.has_value() && quant_mode.value() < 0) {
    MS_EXCEPTION(ValueError) << "For op [" << op_name << "], quant_mode must >= 0, but got " << quant_mode.value();
  }

  // quant_mode:0 , static quant mode, which scale and offset should be 1D tensor;
  if (quant_mode.value() == 0) {
    auto &quant_offset = input_infos[kIndex10];
    const auto &scale_shape = quant_scale->GetShape();
    const auto &offset_shape = quant_offset->GetShape();
    if ((scale_shape.size() != 1) || (offset_shape.size() != 1)) {
      MS_EXCEPTION(ShapeError) << "For op [" << op_name
                               << "], the  shape of scale and offset must be the 1D tensor in static quant scenario, "
                               << "but now get scale dim is " << scale_shape.size() << ", offset dim is "
                               << offset_shape.size();
    }
  } else {
    // dynamic quant, scale must be empty tensor or 2D tensor.
    if (quant_scale->IsNone()) return;
    const auto &scale_shape = quant_scale->GetShape();
    auto constexpr kDim2 = 2;
    if (scale_shape.size() != kDim2) {
      MS_EXCEPTION(ShapeError) << "For op [" << op_name << "], the  scale  must be the 2D tensor or None,"
                               << "but now get scale shape is " << ShapeVectorToStr(scale_shape);
    }
  }
}

ShapeArray MoeInitRoutingQuantV2FuncImpl::GetOutputShapes(int64_t drop_pad_mode_val, int64_t expert_num_val,
                                                          int64_t expert_capacity_val, int64_t h_,
                                                          int64_t active_num_val, int64_t expd_row_idx_dim) const {
  ShapeArray out_shapes;
  auto any_dim = abstract::Shape::kShapeDimAny;
  auto dynamic_quant_scale_val = any_dim;
  if (drop_pad_mode_val) {
    // drop/pad
    (void)out_shapes.emplace_back(std::vector<int64_t>{expert_num_val, expert_capacity_val, h_});
    if ((expert_num_val != any_dim) && (expert_capacity_val != any_dim)) {
      dynamic_quant_scale_val = (expert_num_val * expert_capacity_val);
    }
  } else {
    if (active_num_val == any_dim) {
      (void)out_shapes.emplace_back(std::vector<int64_t>{any_dim, h_});
    } else if (active_num_val != 0) {
      // active
      const int64_t min_active = std::min({active_num_val, expd_row_idx_dim});
      (void)out_shapes.emplace_back(std::vector<int64_t>{min_active, h_});
      if ((active_num_val != any_dim) && (expd_row_idx_dim != any_dim)) {
        dynamic_quant_scale_val = min_active;
      }
    } else {
      // dropless
      (void)out_shapes.emplace_back(std::vector<int64_t>{expd_row_idx_dim, h_});
      if (expd_row_idx_dim != any_dim) {
        dynamic_quant_scale_val = expd_row_idx_dim;
      }
    }
  }

  (void)out_shapes.emplace_back(std::vector<int64_t>{expd_row_idx_dim});
  (void)out_shapes.emplace_back(std::vector<int64_t>{expert_num_val});
  (void)out_shapes.emplace_back(std::vector<int64_t>{expert_num_val});
  (void)out_shapes.emplace_back(std::vector<int64_t>{dynamic_quant_scale_val});
  return out_shapes;
}

ShapeArray MoeInitRoutingQuantV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &x_info = input_infos[kIndex0];
  auto &expert_idx_info = input_infos[kIndex1];

  auto quant_mode = input_infos[kIndex8]->GetScalarValue<int64_t>();
  if (!quant_mode.has_value()) {
    MS_EXCEPTION(ValueError) << "For op [" << op_name << "], quant_mode must have value, but got nothing.";
  }
  auto any_dim = abstract::Shape::kShapeDimAny;
  auto any_shape = abstract::TensorShape::kShapeRankAny;
  if (x_info->IsDynamicRank() || expert_idx_info->IsDynamicRank()) {
    return {{any_shape}, {any_dim}, {any_dim}, {any_dim}, {any_dim}};
  }
  CheckInputs(primitive, input_infos);

  const auto &x_shp = x_info->GetShape();
  const auto &expert_idx_shp = expert_idx_info->GetShape();
  auto num_rows = x_shp[kIndex0];
  auto h = x_shp[kIndex1];
  auto k = expert_idx_shp[kIndex1];
  ShapeValueDType h_ = h == any_dim ? any_dim : h;
  ShapeValueDType expd_row_idx_dim = (num_rows == any_dim || k == any_dim) ? any_dim : num_rows * k;

  auto active_num = input_infos[kIndex2]->GetScalarValue<int64_t>();
  auto expert_capacity = input_infos[kIndex3]->GetScalarValue<int64_t>();
  auto expert_num = input_infos[kIndex4]->GetScalarValue<int64_t>();
  auto drop_pad_mode = input_infos[kIndex5]->GetScalarValue<int64_t>();

  auto active_num_val = active_num.has_value() ? active_num.value() : any_dim;
  auto expert_capacity_val = expert_capacity.has_value() ? expert_capacity.value() : any_dim;
  auto expert_num_val = expert_num.has_value() ? expert_num.value() : any_dim;
  auto drop_pad_mode_val = drop_pad_mode.has_value() ? drop_pad_mode.value() : any_dim;
  if (drop_pad_mode_val == any_dim) {
    return {{any_shape}, {expd_row_idx_dim}, {expert_num_val}, {expert_num_val}, {any_dim}};
  }

  return GetOutputShapes(drop_pad_mode_val, expert_num_val, expert_capacity_val, h_, active_num_val, expd_row_idx_dim);
}

TypeIdList MoeInitRoutingQuantV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  const std::set<TypeId> support_tensor_types = {kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32};
  const std::set<TypeId> support_idx_types = {kNumberTypeInt32};
  const std::set<TypeId> support_quant_types = {kNumberTypeFloat32};
  auto x_type = input_infos[kIndex0]->GetType();
  auto idx_type = input_infos[kIndex1]->GetType();
  CheckType(support_tensor_types, x_type, op_name, "x");
  CheckType(support_idx_types, idx_type, op_name, "expert_idx");
  auto quant_mode = input_infos[kIndex8]->GetScalarValue<int64_t>();
  if (!quant_mode.has_value()) {
    MS_EXCEPTION(ValueError) << "For op [" << op_name << "], quant_mode must be 0 or 1, but got nothing.";
  }
  if ((quant_mode.value() != 0) && (quant_mode.value() != 1)) {
    MS_EXCEPTION(ValueError) << "For op [" << op_name << "], quant_mode must be 0 or 1, but got " << quant_mode.value();
  }
  if (quant_mode.value() == 0) {
    // static quant mode;
    auto scale_type = input_infos[kIndex9]->GetType();
    auto offset_type = input_infos[kIndex10]->GetType();
    CheckType(support_quant_types, scale_type, op_name, "scale");
    CheckType(support_quant_types, offset_type, op_name, "offset");
    return {kNumberTypeInt8, idx_type, idx_type, idx_type, kNumberTypeFloat32};
  }

  // dynamic quant mode;
  if (!input_infos[kIndex9]->IsNone()) {
    auto scale_type = input_infos[kIndex9]->GetType();
    CheckType(support_quant_types, scale_type, op_name, "scale");
  }

  return {kNumberTypeInt8, idx_type, idx_type, idx_type, kNumberTypeFloat32};
}
}  // namespace ops
}  // namespace mindspore
