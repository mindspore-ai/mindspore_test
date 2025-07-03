/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <map>
#include <string>
#include "infer/ops_func_impl/mul.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_constants.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore::ops {
BaseShapePtr MulFuncImpl::InferShape(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr MulFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->GetType());
  (void)types.emplace("y", input_args[kInputIndex1]->GetType());
  return CheckAndConvertUtils::CheckMathBinaryOpTensorType(types, common_valid_types_with_complex_and_bool,
                                                           primitive->name());
}

TypePtrList MulFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  const auto &y_tensor = input_values[kIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(y_tensor);
  const auto &x_dtype = x_tensor->Dtype();
  const auto &y_dtype = y_tensor->Dtype();

  if (MS_UNLIKELY(x_dtype->type_id() != y_dtype->type_id())) {
    auto output_dtype = PromoteType(x_dtype, y_dtype, primitive->name());
    MS_LOG(DEBUG) << "For Mul, 'x' and 'y' have different dtypes with " << TypeIdToString(x_dtype->type_id()) << " and "
                  << TypeIdToString(y_dtype->type_id()) << ", output dtype will be promoteType "
                  << TypeIdToString(output_dtype->type_id()) << ". This happens when data_group is invalid.";
    return {output_dtype};
  }
  return {x_tensor->Dtype()};
}
ShapeArray MulFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {BroadCastInferShape(primitive->name(), input_values)};
}
REGISTER_SIMPLE_INFER(kNameMul, MulFuncImpl)
}  // namespace mindspore::ops
