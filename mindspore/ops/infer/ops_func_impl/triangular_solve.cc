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

#include "infer/ops_func_impl/triangular_solve.h"
#include <vector>
#include <string>
#include <set>
#include "ops_utils/op_utils.h"
#include "infer/ops_func_impl/reduce_arithmetic.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray TriangularSolveFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  const auto &prim_name = primitive->name();
  const auto &input_b = input_infos[kInputIndex0];
  const auto &input_A = input_infos[kInputIndex1];
  const auto input_b_shape = input_infos[kInputIndex0]->GetShape();
  const auto input_A_shape = input_infos[kInputIndex1]->GetShape();
  if (input_b->IsDynamicRank() || input_A->IsDynamicRank()) {
    return {{abstract::Shape::kShapeRankAny}, {abstract::Shape::kShapeRankAny}};
  }
  const int64_t b_num_dims = SizeToLong(input_b_shape.size());
  const int64_t A_num_dims = SizeToLong(input_A_shape.size());
  CheckAndConvertUtils::CheckInRange("dim of input b", b_num_dims, kIncludeBoth, {2, 6}, prim_name);
  CheckAndConvertUtils::CheckInRange("dim of input A", A_num_dims, kIncludeBoth, {2, 6}, prim_name);
  auto input_b_batch_shape = ShapeVector(input_b_shape.begin(), input_b_shape.end() - kDim2);
  auto input_A_batch_shape = ShapeVector(input_A_shape.begin(), input_A_shape.end() - kDim2);
  auto broadcast_shape = CalBroadCastShape(input_b_batch_shape, input_A_batch_shape, prim_name, "b", "A");
  auto b_broadcast_shape = broadcast_shape;
  auto A_broadcast_shape = broadcast_shape;
  b_broadcast_shape.insert(b_broadcast_shape.end(), input_b_shape.end() - kDim2, input_b_shape.end());
  A_broadcast_shape.insert(A_broadcast_shape.end(), input_A_shape.end() - kDim2, input_A_shape.end());
  return {b_broadcast_shape, A_broadcast_shape};
}

std::vector<TypeId> TriangularSolveFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  const auto type = input_infos[kInputIndex0]->GetType();
  return {type, type};
}
}  // namespace ops
}  // namespace mindspore
