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

#include "infer/ops_func_impl/moe_init_routing_v2.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kInputRankSize = 2;
}  // namespace
ShapeArray MoeInitRoutingV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto &x_info = input_infos[kIndex0];
  auto &expert_idx_info = input_infos[kIndex1];

  if (x_info->IsDynamicRank() || expert_idx_info->IsDynamicRank()) {
    auto any_shape = abstract::TensorShape::kShapeRankAny;
    return {{any_shape}, {any_shape}, {any_shape}, {any_shape}};
  }
  CheckRank(x_info, kInputRankSize, op_name, "x");
  CheckRank(expert_idx_info, kInputRankSize, op_name, "expert_idx");
  const auto &x_shp = x_info->GetShape();
  const auto &expert_idx_shp = expert_idx_info->GetShape();
  if (x_shp.front() != expert_idx_shp.front()) {
    MS_EXCEPTION(ShapeError) << "For op [" << op_name << "], the first dim of x and expert_idx must be the same.";
  }
  auto any_dim = abstract::Shape::kShapeDimAny;
  auto num_rows = x_shp[kIndex0];
  auto h = x_shp[kIndex1];
  auto k = expert_idx_shp[kIndex1];
  auto active_num = input_infos[kIndex2]->GetScalarValueWithCheck<int64_t>();
  auto expert_capacity = input_infos[kIndex3]->GetScalarValueWithCheck<int64_t>();
  auto expert_num = input_infos[kIndex4]->GetScalarValueWithCheck<int64_t>();
  auto drop_pad_mode = input_infos[kIndex5]->GetScalarValueWithCheck<int64_t>();
  ShapeValueDType h_ = h == any_dim ? any_dim : h;
  ShapeValueDType expd_row_idx_dim = (num_rows == any_dim || k == any_dim) ? any_dim : num_rows * k;

  ShapeArray out_shapes;
  if (drop_pad_mode) {
    // drop/pad
    (void)out_shapes.emplace_back(std::vector<int64_t>{expert_num, expert_capacity, h_});
  } else {
    // dropless/active
    // active
    if (active_num) {
      (void)out_shapes.emplace_back(std::vector<int64_t>{std::min({active_num, expd_row_idx_dim}), h_});
    } else {
      // dropless
      (void)out_shapes.emplace_back(std::vector<int64_t>{expd_row_idx_dim, h_});
    }
  }
  (void)out_shapes.emplace_back(std::vector<int64_t>{expd_row_idx_dim});
  (void)out_shapes.emplace_back(std::vector<int64_t>{expert_num});
  (void)out_shapes.emplace_back(std::vector<int64_t>{expert_num});
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
