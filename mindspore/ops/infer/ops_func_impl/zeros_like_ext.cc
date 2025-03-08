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

#include "infer/ops_func_impl/zeros_like_ext.h"
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_constants.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_z.h"

namespace mindspore {
namespace ops {
TypePtr ZerosLikeExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto dtype_type = input_args[kInputIndex1]->GetType();
  if (dtype_type->isa<TypeNone>()) {
    return input_args[kInputIndex0]->GetType();
  }
  auto dtype_ptr = input_args[kInputIndex1]->GetValue();
  if (!dtype_ptr->isa<Int64Imm>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', 'dtype' must be a TypeId, but got an invalid type: " << dtype_ptr->ToString() << ".";
  }
  auto val = GetValue<int64_t>(dtype_ptr);
  auto output_type = TypeIdToType(static_cast<TypeId>(val));
  return std::make_shared<TensorType>(output_type);
}

TypePtrList ZerosLikeExtFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  auto prim_name = primitive->name();
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  auto dtype = input_values[kIndex1];
  if (dtype->isa<None>()) {
    return {x_tensor->Dtype()};
  }
  if (!dtype->isa<Int64Imm>()) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', 'dtype' must be a TypeId, but got an invalid type: " << dtype->ToString() << ".";
  }
  const auto &dtype_scalar = dtype->cast<Int64ImmPtr>();
  MS_EXCEPTION_IF_NULL(dtype_scalar);
  auto type_id = static_cast<TypeId>(dtype_scalar->value());
  return {TypeIdToType(type_id)};
}

REGISTER_SIMPLE_INFER(kNameZerosLikeExt, ZerosLikeExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
