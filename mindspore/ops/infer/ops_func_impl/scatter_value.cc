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

#include "infer/ops_func_impl/scatter_value.h"
#include <memory>
#include <utility>
#include "op_def/auto_generate/gen_ops_name.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
BaseShapePtr ScatterValueFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto indices_shape_ptr = input_args[kIndex2]->GetShape();
  const auto &input_shape = input_shape_ptr->GetShapeVector();
  const auto &indices_shape = indices_shape_ptr->GetShapeVector();
  MS_EXCEPTION_IF_CHECK_FAIL(!IsShapeNone(input_shape),
                             "For ScatterValue, [input] got empty tensor, which is not allowed.");
  if (IsDynamicRank(input_shape) && !IsDynamicRank(indices_shape)) {
    auto rank = indices_shape.size();
    ShapeVector output_shape(rank, abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::TensorShape>(std::move(output_shape));
  }
  return input_shape_ptr->Clone();
}

TypePtr ScatterValueFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}

ShapeArray ScatterValueFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &input_shape = input_tensor->shape();
  MS_EXCEPTION_IF_CHECK_FAIL(!IsShapeNone(input_shape),
                             "For ScatterValue, [input] got empty tensor, which is not allowed.");
  return {input_shape};
}

TypePtrList ScatterValueFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &input_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &input_type = input_tensor->Dtype();
  return {input_type};
}

REGISTER_SIMPLE_INFER(kNameScatterValue, ScatterValueFuncImpl)
}  // namespace ops
}  // namespace mindspore
