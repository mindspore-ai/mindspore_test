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

#include "infer/ops_func_impl/argsort.h"
#include <utility>
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ir/dtype.h"
#include "mindspore/ops/op_def/op_name.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ArgSortFuncImpl::InferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_vec = input_args[kInputIndex0]->GetShape()->GetShapeVector();
  return std::make_shared<abstract::TensorShape>(x_shape_vec);
}

ShapeArray ArgSortFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_shape_vec = x_tensor->shape();
  ShapeVector output_shape(x_shape_vec);
  return {output_shape};
}

TypePtr ArgSortFuncImpl::InferType(const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto type = kInt64;
  return std::make_shared<TensorType>(type);
}

TypePtrList ArgSortFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  return {kInt64};
}

}  // namespace ops
}  // namespace mindspore
