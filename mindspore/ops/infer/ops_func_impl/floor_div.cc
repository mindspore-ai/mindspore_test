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

#include "infer/ops_func_impl/floor_div.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
ShapeArray FloorDivFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto prim_name = primitive->name();
  auto input_shape = input_infos[kInputIndex0]->GetShape();
  auto other_shape = input_infos[kInputIndex1]->GetShape();
  auto output_shape = CalBroadCastShape(input_shape, other_shape, prim_name, "input", "other");
  return {output_shape};
}

std::vector<TypeId> FloorDivFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  const auto input_type = input_infos[kInputIndex0]->GetType();
  const auto other_type = input_infos[kInputIndex1]->GetType();
  return {PromoteType(input_type, other_type, primitive->name())};
}
}  // namespace ops
}  // namespace mindspore
