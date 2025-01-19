/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/inplace_mul.h"
#include <memory>
#include "op_def/op_name.h"
#include "mindspore/core/include/mindapi/base/type_id.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore::ops {
static inline bool InplaceMulIsIntegral(TypeId type) {
  return common_integral_type_ids.find(type) != common_integral_type_ids.end();
}

static inline bool InplaceMulIsFloat(TypeId type) {
  return common_float_type_ids.find(type) != common_float_type_ids.end();
}

static inline bool InplaceMulIsComplex(TypeId type) {
  return type == kNumberTypeComplex64 || type == kNumberTypeComplex128;
}

static inline bool InplaceMulCanCast(TypeId from, TypeId to) {
  if (InplaceMulIsComplex(from) && !InplaceMulIsComplex(to)) {
    return false;
  }
  if (InplaceMulIsFloat(from) && InplaceMulIsIntegral(to)) {
    return false;
  }
  if (from != kNumberTypeBool && to == kNumberTypeBool) {
    return false;
  }
  return true;
}

ShapeArray InplaceMulFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}

std::vector<TypeId> InplaceMulFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kInputIndex0]->GetType();
  auto other_type = input_infos[kInputIndex1]->GetType();
  // Align Pytorch's logic on arithmetic operations.
  // For details, please refer to "torch.dtype".
  if (MS_UNLIKELY(!InplaceMulCanCast(other_type, input_type))) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name() << ", other type " << TypeIdToString(other_type)
                            << " can't be cast to the desired output type " << TypeIdToString(input_type);
  }
  return {input_type};
}
}  // namespace mindspore::ops
