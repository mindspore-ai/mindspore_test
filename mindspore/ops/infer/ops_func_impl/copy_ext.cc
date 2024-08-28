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

#include "infer/ops_func_impl/copy_ext.h"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ir/dtype.h"
#include "op_def/op_name.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
BaseShapePtr CopyExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape = input_args[kIndex0]->GetShape();
  return input_shape->Clone();
}

TypePtr CopyExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType();
  return input_type;
}
TypePtrList CopyExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &variable_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(variable_tensor);
  const auto &input_type = variable_tensor->Dtype();
  return {input_type};
}
ShapeArray CopyExtFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &variable_tensor = input_values[kIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(variable_tensor);
  return {variable_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameCopyExt, CopyExtFuncImpl)
}  // namespace mindspore::ops
