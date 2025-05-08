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

#include <map>
#include <string>
#include <set>
#include "infer/ops_func_impl/bitwise_or_tensor.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_constants.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"

namespace mindspore {
namespace ops {
BaseShapePtr BitwiseOrTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  return BroadCastInferShape(primitive->name(), input_args);
}

TypePtr BitwiseOrTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}

TypePtrList BitwiseOrTensorFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &input_type = x_tensor->Dtype();
  return {input_type};
}

ShapeArray BitwiseOrTensorFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {BroadCastInferShape(primitive->name(), input_values)};
}
REGISTER_SIMPLE_INFER(kNameBitwiseOrTensor, BitwiseOrTensorFuncImpl)
}  // namespace ops
}  // namespace mindspore
