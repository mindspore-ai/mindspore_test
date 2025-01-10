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

#include "infer/ops_func_impl/muls.h"
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

constexpr int typeLevelBool = 0;
constexpr int typeLevelInt = 1;
constexpr int typeLevelFloat = 2;
constexpr int typeLevelComplex = 3;

static inline bool IsBoolType(TypeId t) { return t == kNumberTypeBool; }

static inline bool IsIntegralType(TypeId t) {
  return t == kNumberTypeInt8 || t == kNumberTypeInt16 || t == kNumberTypeInt32 || t == kNumberTypeInt64 ||
         t == kNumberTypeUInt8 || t == kNumberTypeUInt16 || t == kNumberTypeUInt32 || t == kNumberTypeUInt64;
}

static inline bool IsFloatingType(TypeId t) {
  return t == kNumberTypeFloat16 || t == kNumberTypeFloat32 || t == kNumberTypeFloat64 || t == kNumberTypeBFloat16;
}

static inline int TypeToLevel(TypeId t) {
  if (IsBoolType(t)) {
    return typeLevelBool;
  } else if (IsIntegralType(t)) {
    return typeLevelInt;
  } else if (IsFloatingType(t)) {
    return typeLevelFloat;
  } else {
    return typeLevelComplex;
  }
}

ShapeArray MulsFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}

std::vector<TypeId> MulsFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  TypeId input_type_id = input_infos[kInputIndex0]->GetType();
  TypeId other_type_id = input_infos[kInputIndex1]->GetType();

  auto promote_type_id = (TypeToLevel(input_type_id) < TypeToLevel(other_type_id)) ? other_type_id : input_type_id;
  return {promote_type_id};
}
}  // namespace mindspore::ops
