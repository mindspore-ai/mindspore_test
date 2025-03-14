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

#include "infer/ops_func_impl/addr.h"
#include <vector>
#include <string>
#include <set>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray AddrFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &op_name = primitive->name();
  const auto &input = input_infos[kInputIndex0];
  MS_EXCEPTION_IF_NULL(input);
  const auto &vec1 = input_infos[kInputIndex1];
  MS_EXCEPTION_IF_NULL(vec1);
  const auto &vec2 = input_infos[kInputIndex2];
  MS_EXCEPTION_IF_NULL(vec2);
  const auto input_shape = input->GetShape();
  const auto vec1_shape = vec1->GetShape();
  const auto vec2_shape = vec2->GetShape();

  int64_t N;
  int64_t M;
  if (vec1->IsDynamic()) {
    N = abstract::Shape::kShapeDimAny;
  } else {
    CheckAndConvertUtils::CheckInteger("vec1 shape size", SizeToLong(vec1_shape.size()), kEqual, kDim1, op_name);
    N = vec1_shape[0];
  }

  if (vec2->IsDynamic()) {
    M = abstract::Shape::kShapeDimAny;
  } else {
    CheckAndConvertUtils::CheckInteger("vec2 shape size", SizeToLong(vec2_shape.size()), kEqual, kDim1, op_name);
    M = vec2_shape[0];
  }

  if (!input->IsDynamicRank()) {
    CheckAndConvertUtils::CheckInRange("dim of input", SizeToLong(input_shape.size()), kIncludeBoth, {kDim1, kDim2},
                                       op_name);
  }

  ShapeVector ret_shape = {N, M};
  CalBroadCastShape(input_shape, ret_shape, op_name, "input", "vec1 ⊗ vec2");
  return {ret_shape};
}

std::vector<TypeId> AddrFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const std::set<TypeId> valid_dtype_set = {
    kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeInt32,   kNumberTypeInt64, kNumberTypeInt16,
    kNumberTypeInt8,    kNumberTypeUInt8,   kNumberTypeFloat64, kNumberTypeBool,  kNumberTypeBFloat16};
  const auto &prim_name = primitive->name();
  const auto input_type = input_infos[kInputIndex0]->GetType();
  const auto vec1_type = input_infos[kInputIndex1]->GetType();
  const auto vec2_type = input_infos[kInputIndex2]->GetType();
  if (vec1_type != vec2_type) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name() << ", the vec1 type: " << vec1_type
                            << " should be the same as vec2 type: " << vec2_type;
  }
  const auto beta_type = input_infos[kInputIndex3]->GetType();
  const auto alpha_type = input_infos[kInputIndex4]->GetType();
  CheckAndConvertUtils::CheckTypeIdValid("input", input_type, valid_dtype_set, prim_name);
  CheckAndConvertUtils::CheckTypeIdValid("vec1", vec1_type, valid_dtype_set, prim_name);
  CheckAndConvertUtils::CheckTypeIdValid("vec2", vec2_type, valid_dtype_set, prim_name);
  CheckAndConvertUtils::CheckTypeIdValid("beta", beta_type, valid_dtype_set, prim_name);
  CheckAndConvertUtils::CheckTypeIdValid("alpha", alpha_type, valid_dtype_set, prim_name);
  return {input_type};
}
}  // namespace ops
}  // namespace mindspore
