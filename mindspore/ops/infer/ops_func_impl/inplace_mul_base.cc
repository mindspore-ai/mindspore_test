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

#include "infer/ops_func_impl/inplace_mul_base.h"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <stdexcept>
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ir/dtype.h"
#include "op_def/op_name.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/muls.h"

namespace mindspore::ops {
ShapeArray InplaceMulBase::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}

std::vector<TypeId> InplaceMulBase::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  TypeId input_type_id = input_infos[kInputIndex0]->GetType();
  TypeId other_type_id = input_infos[kInputIndex1]->GetType();

  int input_level = TypeToLevel(input_type_id);
  int other_level = TypeToLevel(other_type_id);

  if (input_level < other_level) {
    MS_EXCEPTION(ValueError) << "For Inplace operator " << primitive->name()
                             << ", tensor.dtype=" << TypeIdToString(input_type_id)
                             << " should have higher category, but got other.dtype=" << TypeIdToString(other_type_id);
    throw std::invalid_argument("Dtype Bad");
  }
  return {input_type_id};
}
}  // namespace mindspore::ops
