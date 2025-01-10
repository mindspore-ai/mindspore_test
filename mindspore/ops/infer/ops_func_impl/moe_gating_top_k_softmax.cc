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

#include "infer/ops_func_impl/moe_gating_top_k_softmax.h"
#include <memory>
#include <set>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr MoeGatingTopKSoftmaxFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  const auto &x_base_shape = input_args[kInputIndex0]->GetShape();
  auto x_shape = x_base_shape->GetShapeVector();
  // input x dynamic rank
  if (x_base_shape->IsDimUnknown()) {
    ShapeVector dyrank_shape{abstract::TensorShape::kShapeRankAny};
    auto dy_rank_out = std::make_shared<abstract::TensorShape>(dyrank_shape);
    return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList({dy_rank_out, dy_rank_out, dy_rank_out}));
  }
  auto x_rank = x_shape.size();
  if (x_rank != kDim2 && x_rank != kDim3) {
    MS_LOG(EXCEPTION) << "For '" << prim_name << "', input 'X' must be 2D or 3D, but got:" << x_rank;
  }
  auto out_shape_vec = x_shape;
  auto k_scalar = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  if (k_scalar.has_value()) {
    auto k = k_scalar.value();
    out_shape_vec[x_rank - 1] = k;
  } else {
    out_shape_vec[x_rank - 1] = abstract::Shape::kShapeDimAny;
  }
  auto out_shape = std::make_shared<abstract::TensorShape>(out_shape_vec);
  return std::make_shared<abstract::TupleShape>(abstract::BaseShapePtrList({out_shape, out_shape, out_shape}));
}

TypePtr MoeGatingTopKSoftmaxFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = primitive->name();
  const std::set<TypePtr> tensor_valid_types = {kFloat16, kFloat32, kBFloat16};
  const auto &x_type = input_args[kInputIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, tensor_valid_types, prim_name);
  const auto &idx_type = std::make_shared<TensorType>(kInt32);
  return std::make_shared<Tuple>(std::vector<TypePtr>{x_type, idx_type, idx_type});
}
}  // namespace ops
}  // namespace mindspore
