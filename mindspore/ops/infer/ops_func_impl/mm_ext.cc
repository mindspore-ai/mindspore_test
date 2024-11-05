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

#include "infer/ops_func_impl/mm_ext.h"
#include <map>
#include <vector>
#include <string>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
ShapeArray MmFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &input = input_infos[kInputIndex0];
  const auto &mat2 = input_infos[kInputIndex1];
  ShapeVector out_shape;
  if (MS_UNLIKELY(input->IsDynamicRank())) {
    out_shape.emplace_back(abstract::Shape::kShapeDimAny);
  } else {
    const auto &input_shape = input->GetShape();
    if (input_shape.size() != kDim2) {
      MS_EXCEPTION(ValueError) << "For 'Mm' op, input's rank should be 2, but got " << input_shape.size();
    }
    out_shape.emplace_back(input_shape[0]);
  }
  if (MS_UNLIKELY(mat2->IsDynamicRank())) {
    out_shape.emplace_back(abstract::Shape::kShapeDimAny);
  } else {
    const auto &mat2_shape = mat2->GetShape();
    if (mat2_shape.size() != kDim2) {
      MS_EXCEPTION(ValueError) << "For 'Mm' op, mat2's rank should be 2, but got " << mat2_shape.size();
    }
    out_shape.emplace_back(mat2_shape[1]);
  }
  return {out_shape};
}

std::vector<TypeId> MmFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &input = input_infos[kInputIndex0];
  return {input->GetType()};
}
}  // namespace ops
}  // namespace mindspore
