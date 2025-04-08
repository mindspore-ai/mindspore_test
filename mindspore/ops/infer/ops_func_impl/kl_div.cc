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
#include "infer/ops_func_impl/kl_div.h"
#include <set>
#include <memory>
#include <vector>
#include <algorithm>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
bool IsKLDivBroadcastable(const std::vector<int64_t> &x_shape, const std::vector<int64_t> &y_shape) {
  if (x_shape.size() < y_shape.size()) {
    return IsKLDivBroadcastable(y_shape, x_shape);
  }

  if (x_shape == y_shape || IsDynamicRank(x_shape) || IsDynamicRank(y_shape)) {
    return true;
  }

  auto miss = x_shape.size() - y_shape.size();
  for (size_t i = 0; i < y_shape.size(); i++) {
    if (x_shape[miss + i] == y_shape[i]) {
      continue;
    }
    if (x_shape[miss + i] == abstract::TensorShape::kShapeDimAny || x_shape[miss + i] == 1) {
      continue;
    }
    if (y_shape[i] == abstract::TensorShape::kShapeDimAny || y_shape[i] == 1) {
      continue;
    }
    return false;
  }
  return true;
}

ShapeArray InferBroadcastShape(const ShapeVector &x_shape, const ShapeVector &y_shape) {
  ShapeVector output_shape(x_shape);
  size_t miss = x_shape.size() - y_shape.size();
  for (size_t i = 0; i < y_shape.size(); ++i) {
    if ((x_shape[miss + i] == abstract::TensorShape::kShapeDimAny && y_shape[i] == 1) ||
        (x_shape[miss + i] == 1 && y_shape[i] == abstract::TensorShape::kShapeDimAny)) {
      output_shape[miss + i] = abstract::TensorShape::kShapeDimAny;
    } else {
      output_shape[miss + i] = std::max(x_shape[miss + i], y_shape[i]);
    }
  }
  return {output_shape};
}

ShapeArray InferShapeReductionNone(const ShapeVector &input_shape, const ShapeVector &target_shape) {
  if (IsDynamicRank(input_shape) || IsDynamicRank(target_shape)) {
    return ShapeArray{ShapeVector({abstract::TensorShape::kShapeRankAny})};
  }

  if (input_shape == target_shape) {
    return {input_shape};
  }
  if (input_shape.size() < target_shape.size()) {
    return InferBroadcastShape(target_shape, input_shape);
  }
  return InferBroadcastShape(input_shape, target_shape);
}

ShapeArray KLDivFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &input = input_infos[kInputIndex0];
  const auto &target = input_infos[kInputIndex1];
  const auto &reduction_opt = input_infos[kInputIndex2]->GetScalarValue<int64_t>();
  const auto &input_shape = input->GetShape();
  const auto &target_shape = target->GetShape();

  if (!IsKLDivBroadcastable(input_shape, target_shape)) {
    MS_EXCEPTION(ValueError) << "For primitive[KLDiv]"
                             << ", the shape of 'target' and 'input' should be broadcastable.";
  }
  if (!reduction_opt.has_value()) {
    return ShapeArray{ShapeVector({abstract::TensorShape::kShapeRankAny})};
  }
  auto reduction = static_cast<Reduction>(reduction_opt.value());
  if (reduction != Reduction::NONE) {
    return {{}};
  }

  return InferShapeReductionNone(input_shape, target_shape);
}

std::vector<TypeId> KLDivFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32, kNumberTypeBFloat16};
  const auto &prim_name = primitive->name();
  const auto &input_type = input_infos[kInputIndex0]->GetType();
  const auto &target_type = input_infos[kInputIndex1]->GetType();
  (void)CheckAndConvertUtils::CheckTypeIdValid("input", input_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTypeIdValid("target", target_type, valid_types, prim_name);

  if (input_type != target_type) {
    return {kNumberTypeFloat32};
  }
  return {input_type};
}

}  // namespace ops
}  // namespace mindspore
