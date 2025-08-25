/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/full_like.h"
#include <vector>
#include <string>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray FullLikeFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &input = input_infos[kInputIndex0];
  ShapeVector input_shape = input->GetShape();
  if (input->IsDynamicRank()) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }
  return {input_shape};
}

std::vector<TypeId> FullLikeFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  auto &dtype = input_infos[kInputIndex2];
  TypeId output_type;
  if (!dtype->IsNone()) {
    output_type = static_cast<TypeId>(dtype->GetScalarValueWithCheck<int64_t>());
  } else {
    output_type = input_infos[kInputIndex1]->GetType();
  }
  return {output_type};
}
}  // namespace ops
}  // namespace mindspore
