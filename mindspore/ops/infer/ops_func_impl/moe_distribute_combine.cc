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

#include "infer/ops_func_impl/moe_distribute_combine.h"
#include <memory>
#include <set>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr size_t kExpandXDim = 2;
BaseShapePtr MoeDistributeCombineFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  auto expand_x_shape_ptr = input_args[kInputIndex0]->GetShape();
  const auto &expand_x_shape = expand_x_shape_ptr->GetShapeVector();
  auto expert_ids_shape_ptr = input_args[kInputIndex1]->GetShape();
  const auto &expert_ids_shape = expert_ids_shape_ptr->GetShapeVector();

  ShapeVector out_shape{abstract::TensorShape::kShapeDimAny, abstract::TensorShape::kShapeDimAny};
  if (MS_LIKELY(!IsDynamicRank(expert_ids_shape))) {
    out_shape[kIndex0] = expert_ids_shape[kDim0];
  }
  if (MS_LIKELY(!IsDynamicRank(expand_x_shape))) {
    MS_CHECK_VALUE(expand_x_shape.size() == kExpandXDim,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("rank of expand_x", SizeToLong(expand_x_shape.size()),
                                                               kEqual, kExpandXDim, primitive));
    out_shape[kIndex1] = expand_x_shape[kDim1];
  }
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr MoeDistributeCombineFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = primitive->name();
  const auto output_type = input_args[kInputIndex0]->GetType()->Clone();
  const std::set<TypePtr> valid_types = {kFloat16, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("output", output_type, valid_types, prim_name);
  return output_type;
}
}  // namespace ops
}  // namespace mindspore
