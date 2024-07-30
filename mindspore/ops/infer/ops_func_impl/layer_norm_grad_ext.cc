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
#include "infer/ops_func_impl/layer_norm_grad_ext.h"
#include <memory>
#include "abstract/dshape.h"
#include "op_def/op_name.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr LayerNormGradExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  // Get input tensor shape.
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto x_shape_ptr = input_args[kInputIndex1]->GetShape();
  MS_EXCEPTION_IF_NULL(x_shape_ptr);
  auto x_shape = x_shape_ptr->GetShapeVector();
  auto gamma_shape_ptr = input_args[kInputIndex5]->GetShape();
  std::vector<BaseShapePtr> shapes_list;
  if (IsDynamicRank(x_shape)) {
    auto any_shape =
      std::make_shared<abstract::TensorShape>(std::vector<int64_t>{abstract::TensorShape::kShapeRankAny});
    shapes_list = {any_shape, any_shape, any_shape};
    return std::make_shared<abstract::TupleShape>(shapes_list);
  }
  (void)shapes_list.emplace_back(x_shape_ptr->Clone());
  (void)shapes_list.emplace_back(gamma_shape_ptr->Clone());
  (void)shapes_list.emplace_back(gamma_shape_ptr->Clone());
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr LayerNormGradExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  MS_EXCEPTION_IF_NULL(input_args[kInputIndex1]);
  auto x_type = input_args[kInputIndex1]->GetType();
  MS_EXCEPTION_IF_NULL(x_type);
  std::vector<TypePtr> types_list;
  auto out_type = x_type->Clone();
  types_list = {out_type, out_type, out_type};
  return std::make_shared<Tuple>(types_list);
}

}  // namespace ops
}  // namespace mindspore
