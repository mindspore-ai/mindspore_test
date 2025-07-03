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

#include <string>
#include "infer/ops_func_impl/sub_scalar.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
bool isIntegralType(TypeId t) {
  return t == kNumberTypeInt8 || t == kNumberTypeInt16 || t == kNumberTypeInt32 || t == kNumberTypeInt64 ||
         t == kNumberTypeUInt8 || t == kNumberTypeUInt16 || t == kNumberTypeUInt32 || t == kNumberTypeUInt64;
}

bool isFloatType(TypeId t) {
  return t == kNumberTypeBFloat16 || t == kNumberTypeFloat16 || t == kNumberTypeFloat32 || t == kNumberTypeFloat64;
}

bool isBoolType(TypeId t) { return t == kNumberTypeBool; }

TypeId GetOutputType(const PrimitivePtr &primitive, TypeId input_type, TypeId other_type, TypeId alpha_type) {
  if (isFloatType(alpha_type) && !isFloatType(other_type) && !isFloatType(input_type)) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', floating alpha and int/bool other need floating input, but got "
                             << TypeIdToType(input_type)->ToString();
  }

  if ((isIntegralType(input_type) || isBoolType(input_type)) && isFloatType(other_type)) {
    return kNumberTypeFloat32;
  }

  if (isBoolType(input_type) && isIntegralType(other_type)) {
    return kNumberTypeInt64;
  }

  return input_type;
}
}  // namespace

ShapeArray SubScalarFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[0]->GetShape()};
}

std::vector<TypeId> SubScalarFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  TypeId input_type = input_infos[kInputIndex0]->GetType();
  TypeId other_type = input_infos[kInputIndex1]->GetType();
  TypeId alpha_type = input_infos[kInputIndex2]->GetType();
  auto out_type = GetOutputType(primitive, input_type, other_type, alpha_type);
  return {out_type};
}
}  // namespace ops
}  // namespace mindspore
