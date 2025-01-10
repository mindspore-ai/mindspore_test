/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include <vector>
#include "utils/log_adapter.h"
#include "infer/ops_func_impl/assign.h"
#include "ops_utils/op_constants.h"

namespace mindspore::ops {
ShapeArray AssignFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto prim_name = primitive->name();
  auto &variable = input_infos[kIndex0];
  auto &value = input_infos[kIndex1];
  const auto &variable_shape = variable->GetShape();
  const auto &value_shape = value->GetShape();
  if (variable->IsDynamic()) {
    return {variable_shape};
  }
  if (value->IsDynamic()) {
    return {value_shape};
  }

  if (variable_shape.size() != value_shape.size()) {
    if (variable_shape.size() == 1 && variable_shape[0] == 1 && value_shape.empty()) {
      return {variable_shape};
    } else if (value_shape.size() == 1 && value_shape[0] == 1 && variable_shape.empty()) {
      return {variable_shape};
    } else {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "','value' must have the same rank as 'variable'. But got 'value' rank: "
                               << value_shape.size() << ", 'variable' rank: " << variable_shape.size() << ".";
    }
  }

  for (uint64_t i = 0; i < variable_shape.size(); i++) {
    if (variable_shape[i] != value_shape[i]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "','value' must have the same shape as 'variable'. But got 'value' shape: "
                               << ShapeVectorToStr(value_shape)
                               << ", 'variable' shape: " << ShapeVectorToStr(variable_shape) << ".";
    }
  }

  return {variable_shape};
}

std::vector<TypeId> AssignFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace mindspore::ops
