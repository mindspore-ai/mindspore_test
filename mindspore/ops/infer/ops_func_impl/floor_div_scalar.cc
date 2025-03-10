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

#include "infer/ops_func_impl/floor_div_scalar.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
static inline bool IsIntBoolType(TypeId t) {
  return t == kNumberTypeInt8 || t == kNumberTypeInt16 || t == kNumberTypeInt32 || t == kNumberTypeInt64 ||
         t == kNumberTypeUInt8 || t == kNumberTypeBool;
}

ShapeArray FloorDivScalarFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}

std::vector<TypeId> FloorDivScalarFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  const auto input_type = input_infos[kInputIndex0]->GetType();
  const auto other_type = input_infos[kInputIndex1]->GetType();
  TypeId out_type;

  if (IsIntBoolType(input_type) && other_type == kNumberTypeFloat32) {
    out_type = kNumberTypeFloat32;
  } else if (input_type == kNumberTypeBool && IsIntBoolType(other_type)) {
    out_type = kNumberTypeInt64;
  } else {
    out_type = input_type;
  }

  return {out_type};
}
}  // namespace ops
}  // namespace mindspore
