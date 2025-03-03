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

#include "infer/ops_func_impl/silent_check_v3.h"
#include <utility>
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
ShapeArray SilentCheckV3FuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  ShapeArray output_shapes;
  // avg, input_grad, step
  for (size_t i = kIndex2; i < kIndex5; ++i) {
    auto shape = input_infos[i]->GetShape();
    output_shapes.emplace_back(std::move(shape));
  }
  // result
  output_shapes.emplace_back(std::vector<int64_t>{1});
  return output_shapes;
}

TypeIdList SilentCheckV3FuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  TypeIdList output_types;
  // avg, input_grad, step
  for (size_t i = kIndex2; i < kIndex5; ++i) {
    auto type = input_infos[i]->GetType();
    output_types.push_back(type);
  }
  // result
  output_types.push_back(TypeId::kNumberTypeInt32);
  return output_types;
}
}  // namespace ops
}  // namespace mindspore
