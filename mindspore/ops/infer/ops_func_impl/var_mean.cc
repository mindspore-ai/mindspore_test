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

#include "infer/ops_func_impl/var_mean.h"
#include <vector>
#include <string>
#include <set>
#include "ops_utils/op_utils.h"
#include "infer/ops_func_impl/reduce_arithmetic.h"
#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray VarMeanFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return ReduceGeneralInferShapeV2(primitive, input_infos);
}

std::vector<TypeId> VarMeanFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  const std::set<TypeId> valid_dtype_set = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};
  const auto type = input_infos[kInputIndex0]->GetType();
  const auto &prim_name = primitive->name();
  CheckAndConvertUtils::CheckTypeIdValid("input", type, valid_dtype_set, prim_name);
  return {type, type};
}
}  // namespace ops
}  // namespace mindspore
