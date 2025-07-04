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
#include "infer/ops_func_impl/softshrink.h"

#include "abstract/dshape.h"
#include "op_def/op_name.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_utils.h"
#include "utils/convert_utils_base.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace ops {

BaseShapePtr SoftShrinkFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto in_shape = input_args[kInputIndex0]->GetShape();
  MS_EXCEPTION_IF_NULL(in_shape);

  return in_shape->Clone();
}

TypePtr SoftShrinkFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex0]);
  auto x_type = input_args[kInputIndex0]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  auto tensor_type = x_type->cast<TensorTypePtr>();
  auto real_type = tensor_type->element();
  (void)CheckAndConvertUtils::CheckSubClass("input_x", real_type, valid_types, primitive->name());
  return x_type;
}

TypePtrList SoftShrinkFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kBFloat16};
  const auto &input_type = x_tensor->Dtype();
  (void)CheckAndConvertUtils::CheckSubClass("input_x", input_type, valid_types, primitive->name());
  return {input_type};
}

ShapeArray SoftShrinkFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameSoftShrink, SoftShrinkFuncImpl)
}  // namespace ops
}  // namespace mindspore
