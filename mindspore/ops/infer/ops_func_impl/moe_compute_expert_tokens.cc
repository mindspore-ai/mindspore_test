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

#include "infer/ops_func_impl/moe_compute_expert_tokens.h"
#include <memory>
#include <set>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr MoeComputeExpertTokensFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  auto sorted_experts_shape_ptr = input_args[kInputIndex0]->GetShape();
  const auto &sorted_experts_shape = sorted_experts_shape_ptr->GetShapeVector();

  MS_CHECK_VALUE(sorted_experts_shape.size() == 1,
                 CheckAndConvertUtils::FormatCheckIntegerMsg(
                   "rank of sorted_experts", SizeToLong(sorted_experts_shape.size()), kEqual, 1, primitive));
  auto num_experts_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());

  ShapeVector out_shape{};
  if (!num_experts_opt.has_value()) {
    (void)out_shape.emplace_back(abstract::Shape::kShapeRankAny);
    return std::make_shared<abstract::Shape>(out_shape);
  }

  int64_t num_experts = num_experts_opt.value();
  (void)out_shape.emplace_back(num_experts);
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr MoeComputeExpertTokensFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kInt32};
  const auto &sorted_experts_type = input_args[kInputIndex0]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("sorted_experts", sorted_experts_type, valid_types, prim_name);

  return sorted_experts_type;
}
}  // namespace ops
}  // namespace mindspore
