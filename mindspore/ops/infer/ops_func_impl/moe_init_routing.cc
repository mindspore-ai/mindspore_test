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

#include "infer/ops_func_impl/moe_init_routing.h"
#include <algorithm>
#include <memory>
#include <set>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kmaxsize = 2;

constexpr size_t kx_ = 0;
constexpr size_t krowIdx_ = 1;
constexpr size_t kexpertIdx_ = 2;
constexpr size_t kactiveNum_ = 3;
}  // namespace

BaseShapePtr MoeInitRoutingFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kx_]->GetShape());
  auto row_idx_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[krowIdx_]->GetShape());
  auto expert_idx_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kexpertIdx_]->GetShape());

  auto x_shp = x_shape_map[kShape];
  auto row_idx_shp = row_idx_shape_map[kShape];
  auto expert_idx_shp = expert_idx_shape_map[kShape];
  std::vector<BaseShapePtr> shapes_list;
  if (IsDynamicRank(x_shp) || IsDynamicRank(row_idx_shp) || IsDynamicRank(expert_idx_shp)) {
    auto any_shape =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
    shapes_list = {any_shape, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }

  if (x_shape_map.empty() || x_shape_map[kShape].size() != kmaxsize) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'x' must be a 2D Tensor type, but got:" << input_args[kx_]->ToString();
  }
  if (row_idx_shape_map.empty() || row_idx_shape_map[kShape].size() != kmaxsize) {
    MS_LOG(EXCEPTION) << "For '" << prim_name
                      << "', input 'row_idx' must be a 2D Tensor type, but got:" << input_args[krowIdx_]->ToString();
  }
  if (expert_idx_shape_map.empty() || expert_idx_shape_map[kShape].size() != kmaxsize) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', input 'expert_idx' must be a 2D Tensor type, but got:"
                      << input_args[kexpertIdx_]->ToString();
  }

  const int64_t NUM_ROWS = expert_idx_shp[0];
  const int64_t K = expert_idx_shp[1];
  const int64_t H = x_shp[1];
  auto active_num_scalar = GetScalarValue<int64_t>(input_args[kactiveNum_]->GetValue());
  if (active_num_scalar.has_value()) {
    auto active_num = active_num_scalar.value();
    ShapeVector expandedXOut = {std::min(NUM_ROWS, active_num) * K, H};
    ShapeVector expandedRowIdxOut = {NUM_ROWS * K};
    ShapeVector expandedExpertIdxOut = {NUM_ROWS * K};
    (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(expandedXOut));
    (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(expandedRowIdxOut));
    (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(expandedExpertIdxOut));
  } else {
    ShapeVector expandedXOut = {abstract::Shape::kShapeDimAny, H};
    ShapeVector expandedRowIdxOut = {abstract::Shape::kShapeDimAny};
    ShapeVector expandedExpertIdxOut = {abstract::Shape::kShapeDimAny};
    (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(expandedXOut));
    (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(expandedRowIdxOut));
    (void)shapes_list.emplace_back(std::make_shared<abstract::TensorShape>(expandedExpertIdxOut));
  }
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr MoeInitRoutingFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = primitive->name();
  const std::set<TypePtr> tensor_valid_types = {kFloat16, kFloat32, kBFloat16};
  const std::set<TypePtr> idx_valid_types = {kInt32};
  const auto &infer_type = input_args[kx_]->GetType();
  MS_EXCEPTION_IF_NULL(infer_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", infer_type, tensor_valid_types, prim_name);

  const auto &row_idx_type = input_args[krowIdx_]->GetType();
  MS_EXCEPTION_IF_NULL(row_idx_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("row_idx", row_idx_type, idx_valid_types, prim_name);

  const auto &expert_idx_type = input_args[kexpertIdx_]->GetType();
  MS_EXCEPTION_IF_NULL(expert_idx_type);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("expert_idx", expert_idx_type, idx_valid_types, prim_name);

  std::vector<TypePtr> types_list;
  types_list = {infer_type, expert_idx_type, expert_idx_type};
  return std::make_shared<Tuple>(types_list);
}
}  // namespace ops
}  // namespace mindspore
