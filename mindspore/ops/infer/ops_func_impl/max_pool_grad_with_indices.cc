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

#include "infer/ops_func_impl/max_pool_grad_with_indices.h"
#include <algorithm>
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
TypePtr MaxPoolGradWithIndicesFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kIndex0]->GetType();
  return x_type->Clone();
}

BaseShapePtr MaxPoolGradWithIndicesFuncImpl::InferShape(
  const PrimitivePtr &primitive, const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(
      std::vector<int64_t>{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                           abstract::Shape::kShapeDimAny});
  }
  return std::make_shared<abstract::Shape>(x_shape);
}

}  // namespace ops
}  // namespace mindspore
