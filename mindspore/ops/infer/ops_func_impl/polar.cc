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

#include <string>
#include <map>
#include <utility>
#include <memory>
#include "infer/ops_func_impl/polar.h"
#include "ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
ShapeArray PolarFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const InferInfoPtr &abs = input_infos[kIndex0];
  const InferInfoPtr &angle = input_infos[kIndex1];

  ShapeVector abs_shape = abs->GetShape();
  ShapeVector angle_shape = angle->GetShape();

  if (abs->IsDynamicRank() || angle->IsDynamicRank()) {
    return {ShapeVector{-2}};
  }

  if (abs_shape.size() != angle_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << ", shape of inputs should be the same, but get shape of input[0] : " << abs_shape
                             << " , shape of input[1] : " << angle_shape << " .";
  }

  auto output_shape = abs_shape;
  if (abs->IsDynamic() || angle->IsDynamic()) {
    for (size_t idx = 0; idx < output_shape.size(); ++idx) {
      output_shape[idx] = (output_shape[idx] == -1) ? angle_shape[idx] : output_shape[idx];
    }
    return {output_shape};
  }

  // broadcast
  for (size_t idx = 0; idx < output_shape.size(); ++idx) {
    output_shape[idx] = (output_shape[idx] == 1) ? angle_shape[idx] : output_shape[idx];
    if (output_shape[idx] != angle_shape[idx] && angle_shape[idx] != 1) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << ", shape of inputs should be the same, but get shape of input[0] : " << abs_shape
                               << " , shape of input[1] : " << angle_shape << " .";
    }
  }
  return {output_shape};
}

std::vector<TypeId> PolarFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  std::map<std::string, TypePtr> types;
  auto prim_name = primitive->name();

  auto abs_type = input_infos[kInputIndex0]->GetType();
  auto angle_type = input_infos[kInputIndex1]->GetType();

  (void)types.emplace("abs", std::make_shared<TensorType>(TypeIdToType(abs_type)));
  (void)types.emplace("angle", std::make_shared<TensorType>(TypeIdToType(angle_type)));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, std::set<TypePtr>{kFloat32, kFloat64}, prim_name);

  if (abs_type == kNumberTypeFloat64) {
    return {kNumberTypeComplex128};
  }
  return {kNumberTypeComplex64};
}
}  // namespace ops
}  // namespace mindspore
