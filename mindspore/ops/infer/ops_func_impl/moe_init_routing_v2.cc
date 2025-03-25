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

#include "infer/ops_func_impl/moe_init_routing_v2.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInputRankSize = 2;
}  // namespace

void MoeInitRoutingV2FuncImpl::CheckInputs(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &x_info = input_infos[kIndex0];
  auto &expert_idx_info = input_infos[kIndex1];

  CheckRank(x_info, kInputRankSize, op_name, "x");
  CheckRank(expert_idx_info, kInputRankSize, op_name, "expert_idx");
  const auto &x_shp = x_info->GetShape();
  const auto &expert_idx_shp = expert_idx_info->GetShape();
  if (x_shp.front() != expert_idx_shp.front()) {
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
}

ShapeArray MoeInitRoutingV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  auto &x_info = input_infos[kIndex0];
  auto &expert_idx_info = input_infos[kIndex1];

  auto any_dim = abstract::Shape::kShapeDimAny;
  auto any_shape = abstract::TensorShape::kShapeRankAny;
  if (x_info->IsDynamicRank() || expert_idx_info->IsDynamicRank()) {
    return {{any_shape}, {any_dim}, {any_dim}, {any_dim}};
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
    return {{any_shape}, {expd_row_idx_dim}, {expert_num_val}, {expert_num_val}};
  }

  ShapeArray out_shapes;
  if (drop_pad_mode_val) {
    // drop/pad
    (void)out_shapes.emplace_back(std::vector<int64_t>{expert_num_val, expert_capacity_val, h_});
  } else {
    if (active_num_val == any_dim) {
      (void)out_shapes.emplace_back(std::vector<int64_t>{any_dim, h_});
    } else {
      // active
      if (active_num_val) {
        (void)out_shapes.emplace_back(std::vector<int64_t>{std::min({active_num_val, expd_row_idx_dim}), h_});
      } else {
        // dropless
        (void)out_shapes.emplace_back(std::vector<int64_t>{expd_row_idx_dim, h_});
      }
    }
  }

  (void)out_shapes.emplace_back(std::vector<int64_t>{expd_row_idx_dim});
  (void)out_shapes.emplace_back(std::vector<int64_t>{expert_num_val});
  (void)out_shapes.emplace_back(std::vector<int64_t>{expert_num_val});
  return out_shapes;
}

TypeIdList MoeInitRoutingV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  const std::set<TypeId> support_tensor_types = {kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32};
  const std::set<TypeId> support_idx_types = {kNumberTypeInt32};
  auto x_type = input_infos[kIndex0]->GetType();
  auto idx_type = input_infos[kIndex1]->GetType();
  CheckType(support_tensor_types, x_type, op_name, "x");
  CheckType(support_idx_types, idx_type, op_name, "expert_idx");

  return {x_type, idx_type, idx_type, idx_type};
}
}  // namespace ops
}  // namespace mindspore
