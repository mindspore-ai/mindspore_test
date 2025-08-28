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

#include "infer/ops_func_impl/addmv.h"
#include <vector>
#include <string>
#include <set>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray AddmvFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  const auto &input = input_infos[kInputIndex0];
  MS_EXCEPTION_IF_NULL(input);
  const auto &mat = input_infos[kInputIndex1];
  MS_EXCEPTION_IF_NULL(mat);
  const auto &vec = input_infos[kInputIndex2];
  MS_EXCEPTION_IF_NULL(vec);
  const auto input_shape = input->GetShape();
  const auto mat_shape = mat->GetShape();
  const auto vec_shape = vec->GetShape();

  if (mat->IsDynamicRank() || vec->IsDynamicRank()) {
    ShapeVector ret_shape = {abstract::Shape::kShapeDimAny};
    return {ret_shape};
  }

  bool dynamic_shape = mat->IsDynamic() || vec->IsDynamic();
  if (!dynamic_shape) {
    if (mat_shape.size() != kDim2) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the input 'mat' must be a 2D dimensional Tensor, but got " << mat_shape.size()
                               << "D shape " << mat_shape;
    }
    if (vec_shape.size() != kDim1) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the input 'vec' must be a 1D dimensional Tensor, but got " << vec_shape.size()
                               << "D shape " << vec_shape;
    }
    int64_t mat_col = mat_shape[kIndex1];
    int64_t vec_row = vec_shape[kIndex0];
    if (mat_col != vec_row) {
      MS_EXCEPTION(ValueError)
        << "For " << primitive->name()
        << ", the elements of the input 'mat' should be same as the elements of the input 'vec', with input shape "
        << mat_shape << ", other shape " << vec_shape;
    }
  }
  ShapeVector ret_shape = {mat_shape[0]};
  auto broadcast_shape = CalBroadCastShape(input_shape, ret_shape, op_name, "input", "mat@vec");
  return {broadcast_shape};
}

std::vector<TypeId> AddmvFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const std::set<TypeId> valid_dtype_set1 = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeInt32,
                                             kNumberTypeInt64,   kNumberTypeInt16,   kNumberTypeInt8,
                                             kNumberTypeUInt8,   kNumberTypeFloat64, kNumberTypeBool};
  const auto input_type = input_infos[kInputIndex0]->GetType();
  const auto mat_type = input_infos[kInputIndex1]->GetType();
  const auto vec_type = input_infos[kInputIndex2]->GetType();
  const auto &prim_name = primitive->name();
  CheckAndConvertUtils::CheckTypeIdValid("input", input_type, valid_dtype_set1, prim_name);
  CheckAndConvertUtils::CheckTypeIdValid("mat", mat_type, valid_dtype_set1, prim_name);
  CheckAndConvertUtils::CheckTypeIdValid("vec", vec_type, valid_dtype_set1, prim_name);
  const std::set<TypeId> valid_dtype_set2 = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeFloat64,
                                             kNumberTypeInt8,    kNumberTypeInt16,   kNumberTypeInt32,
                                             kNumberTypeUInt8,   kNumberTypeInt64};
  const auto beta_type = input_infos[kInputIndex3]->GetType();
  const auto alpha_type = input_infos[kInputIndex4]->GetType();
  CheckAndConvertUtils::CheckTypeIdValid("beta", beta_type, valid_dtype_set2, prim_name);
  CheckAndConvertUtils::CheckTypeIdValid("alpha", alpha_type, valid_dtype_set2, prim_name);
  return {input_type};
}
}  // namespace ops
}  // namespace mindspore
