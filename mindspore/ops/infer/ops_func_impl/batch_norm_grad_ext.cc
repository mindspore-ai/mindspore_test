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
#include "infer/ops_func_impl/batch_norm_grad_ext.h"
#include <memory>
#include "abstract/dshape.h"
#include "mindspore/ops/op_def/op_name.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr BatchNormGradExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  auto x_shape_ptr = input_args[kInputIndex1]->GetShape();
  if (input_args[kInputIndex2]->GetType()->isa<TypeNone>()) {
    const auto &x_shape = x_shape_ptr->GetShapeVector();
    ShapeVector weight_shape = {x_shape[1]};
    ShapeVector bias_shape = {x_shape[1]};
    auto weight_shape_ptr = std::make_shared<abstract::TensorShape>(weight_shape);
    auto bias_shape_ptr = std::make_shared<abstract::TensorShape>(bias_shape);
    std::vector<BaseShapePtr> shapes_list{x_shape_ptr->Clone(), weight_shape_ptr, bias_shape_ptr};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }
  auto weight_shape_ptr = input_args[kInputIndex2]->GetShape();
  std::vector<BaseShapePtr> shapes_list{x_shape_ptr->Clone(), weight_shape_ptr->Clone(), weight_shape_ptr->Clone()};
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr BatchNormGradExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto dy_type = input_args[kInputIndex0]->GetType();
  std::vector<TypePtr> types_list;
  types_list = {dy_type, kFloat32, kFloat32};
  return std::make_shared<Tuple>(types_list);
}

}  // namespace ops
}  // namespace mindspore
