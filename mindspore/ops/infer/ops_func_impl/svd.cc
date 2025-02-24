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

#include "infer/ops_func_impl/svd.h"
#include <algorithm>
#include <set>
#include <string>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
int32_t SvdFuncImpl::CheckValidation(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &input = input_infos[kIndex0];
  if (input->IsDynamicRank()) {
    return OP_CHECK_RETRY;
  }
  constexpr size_t kRank = 2;
  auto input_shape = input_infos[kIndex0]->GetShape();
  MS_CHECK_VALUE(input_shape.size() >= kRank, "For " + primitive->name() +
                                                ", the rank of 'input' must be great equal than 2, but get shape: " +
                                                ShapeVectorToString(input_shape));

  auto full_matrices_opt = input_infos[kIndex1]->GetScalarValue<bool>();
  if (!full_matrices_opt.has_value() || input->IsDynamic()) {
    return OP_CHECK_RETRY;
  }

  auto input_rank = SizeToLong(input_shape.size());
  auto m = input_shape[input_rank - kIndex2];
  auto n = input_shape[input_rank - kIndex1];

  if (full_matrices_opt.value() && std::abs(m - n) > 1) {
    MS_LOG(WARNING) << "For " << primitive->name()
                    << ", abs(m - n) > 1 with full_matrices is true may cause error in gradient.";
  }

  return OP_CHECK_SUCCESS;
}

ShapeArray SvdFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &input = input_infos[kIndex0];
  if (input->IsDynamic()) {
    ShapeVector dyn_shape{abstract::TensorShape::kShapeRankAny};
    return {dyn_shape, dyn_shape, dyn_shape};
  }

  auto input_shape = input->GetShape();
  auto input_rank = SizeToLong(input_shape.size());
  auto m = input_shape[input_rank - kIndex2];
  auto n = input_shape[input_rank - kIndex1];
  auto p = std::min(m, n);
  auto s_shape = ShapeVector(input_shape.begin(), input_shape.end() - kIndex2);
  auto u_shape = ShapeVector(input_shape.begin(), input_shape.end() - kIndex2);
  auto v_shape = ShapeVector(input_shape.begin(), input_shape.end() - kIndex2);
  s_shape.emplace_back(p);

  auto compute_uv = input_infos[kIndex2]->GetScalarValue<bool>();
  if (!compute_uv.has_value()) {
    ShapeVector dyn_shape{abstract::TensorShape::kShapeRankAny};
    return {s_shape, dyn_shape, dyn_shape};
  }

  if (compute_uv.value()) {
    u_shape.emplace_back(m);
    v_shape.emplace_back(n);
    auto full_matrices = input_infos[kIndex1]->GetScalarValue<bool>();
    if (!full_matrices.has_value()) {
      u_shape.emplace_back(abstract::TensorShape::kShapeDimAny);
      v_shape.emplace_back(abstract::TensorShape::kShapeDimAny);
    } else {
      u_shape.emplace_back(full_matrices.value() ? m : p);
      v_shape.emplace_back(full_matrices.value() ? n : p);
    }
  } else {
    u_shape = ShapeVector{1};
    v_shape = ShapeVector{1};
  }

  return {s_shape, u_shape, v_shape};
}

std::vector<TypeId> SvdFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const std::set<TypeId> valid_types = {kNumberTypeFloat32, kNumberTypeFloat64, kNumberTypeComplex64,
                                        kNumberTypeComplex128};
  auto input_type = input_infos[kIndex0]->GetType();
  if (valid_types.find(input_type) == valid_types.end()) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name()
                            << ", the dtype of 'input' must be in (float32, float64, complex64, complex128), but got "
                            << TypeIdToString(input_type);
  }
  return {input_type, input_type, input_type};
}
}  // namespace ops
}  // namespace mindspore
