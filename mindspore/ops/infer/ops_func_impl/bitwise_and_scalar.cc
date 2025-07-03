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
#include "infer/ops_func_impl/bitwise_and_scalar.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
std::vector<TypeId> BitwiseAndScalarFuncImpl::InferType(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  auto x_type = input_infos[kIndex0]->GetType();
  auto other_type = input_infos[kIndex1]->GetType();
  if (x_type == kNumberTypeBool && common_integral_type_ids.count(other_type) != 0) {
    return {kNumberTypeInt64};
  }
  return {x_type};
}

ShapeArray BitwiseAndScalarFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetShape()};
}
}  // namespace ops
}  // namespace mindspore
