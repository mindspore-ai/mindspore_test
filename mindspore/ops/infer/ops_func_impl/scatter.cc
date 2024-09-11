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

#include "infer/ops_func_impl/scatter.h"
#include <map>
#include <memory>
#include <string>
#include <utility>
#include "utils/check_convert_utils.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ScatterFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto input_shape_ptr = input_args[kIndex0]->GetShape();
  auto indices_shape_ptr = input_args[kIndex2]->GetShape();
  auto src_shape_ptr = input_args[kIndex3]->GetShape();
  const auto &input_shape = input_shape_ptr->GetShapeVector();
  const auto &indices_shape = indices_shape_ptr->GetShapeVector();
  const auto &src_shape = src_shape_ptr->GetShapeVector();
  MS_EXCEPTION_IF_CHECK_FAIL(!IsShapeNone(input_shape) && !IsShapeNone(src_shape),
                             "For Scatter, [input] or [src] got empty tensor, which is not allowed.");
  if (IsDynamicRank(input_shape)) {
    size_t rank;
    if (!IsDynamicRank(indices_shape)) {
      rank = indices_shape.size();
    } else if (!IsDynamicRank(src_shape)) {
      rank = src_shape.size();
    } else {
      return input_shape_ptr->Clone();
    }
    ShapeVector output_shape(rank, abstract::Shape::kShapeDimAny);
    return std::make_shared<abstract::TensorShape>(std::move(output_shape));
  }
  return input_shape_ptr->Clone();
}

TypePtr ScatterFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto input_type = input_args[kIndex0]->GetType();
  auto src_type = input_args[kIndex3]->GetType();
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input", input_type);
  (void)types.emplace("src", src_type);
  (void)CheckAndConvertUtils::CheckTypeSame(types, primitive->name());

  return input_args[kIndex0]->GetType();
}
}  // namespace ops
}  // namespace mindspore
