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

#include "infer/ops_func_impl/divmods.h"

#include <set>
#include <unordered_set>

#include "mindapi/base/type_id.h"
#include "ops_utils/op_utils.h"

namespace mindspore::ops {
ShapeArray DivModsFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetShape()};
}

std::vector<TypeId> DivModsFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  const std::unordered_set<TypeId> kFloatTypes{kNumberTypeFloat,   kNumberTypeFloat16,   kNumberTypeFloat32,
                                               kNumberTypeFloat64, kNumberTypeBFloat16,  kNumberTypeDouble,
                                               kNumberTypeComplex, kNumberTypeComplex64, kNumberTypeComplex128};
  auto IsFloatType = [&kFloatTypes](TypeId type) { return kFloatTypes.find(type) != kFloatTypes.end(); };

  auto input_type = input_infos[kIndex0]->GetType();
  auto output_type = input_type;
  if (!IsFloatType(input_type)) {
    auto other_type = input_infos[kIndex1]->GetType();
    if (IsFloatType(other_type) || input_infos[kIndex2]->IsNone()) {
      output_type = kNumberTypeFloat32;
    }
  }
  return {output_type};
}
}  // namespace mindspore::ops
