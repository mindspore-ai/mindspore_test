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

#include "infer/ops_func_impl/t_ext.h"
#include <vector>
#include <string>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray TExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto &x = input_infos[kInputIndex0];
  ShapeVector x_shape = x->GetShape();
  if (x->IsDynamicRank()) {
    return {{abstract::TensorShape::kShapeRankAny}};
  }
  auto x_rank = SizeToLong(x_shape.size());
  CheckAndConvertUtils::CheckInRange("rank of input shape", x_rank, kIncludeBoth, {0, 2}, op_name);
  ShapeVector out_shape;
  out_shape.resize(x_rank);
  std::reverse_copy(x_shape.begin(), x_shape.end(), out_shape.begin());
  return {out_shape};
}

std::vector<TypeId> TExtFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto type = input_infos[kInputIndex0]->GetType();
  return {type};
}
}  // namespace ops
}  // namespace mindspore
