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

#include "infer/ops_func_impl/l1_loss_backward_ext.h"
#include <algorithm>
#include <memory>
#include <set>
#include <utility>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"

namespace mindspore {
namespace ops {
TypePtr L1LossBackwardExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<abstract::AbstractBasePtr> &input_args) const {
  auto grad_output_type = input_args[kInputIndex0]->GetType();
  auto input_type = input_args[kInputIndex1]->GetType();
  auto target_type = input_args[kInputIndex2]->GetType();
  MS_EXCEPTION_IF_NULL(grad_output_type);
  MS_EXCEPTION_IF_NULL(input_type);
  MS_EXCEPTION_IF_NULL(target_type);
  auto promote1 = PromoteType(target_type, input_type, primitive->name());
  return std::make_shared<TensorType>(PromoteType(grad_output_type, promote1, primitive->name()));
}

BaseShapePtr L1LossBackwardExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<abstract::AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();

  if (input_args.size() < kSize3) {
    MS_EXCEPTION(ValueError) << "For gradient of L1Loss, the input size is illegal.";
  }
  auto temp_shapes = {input_args[kInputIndex1], input_args[kInputIndex2]};
  return BroadCastInferShape(prim_name, temp_shapes);
}

TypePtrList L1LossBackwardExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const ValuePtrList &input_values) const {
  auto grad_output_type = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>()->Dtype();
  auto input_type = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>()->Dtype();
  auto target_type = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>()->Dtype();
  MS_EXCEPTION_IF_NULL(grad_output_type);
  MS_EXCEPTION_IF_NULL(input_type);
  MS_EXCEPTION_IF_NULL(target_type);
  auto promote1 = PromoteType(target_type, input_type, primitive->name());
  return {PromoteType(grad_output_type, promote1, primitive->name())};
}

ShapeArray L1LossBackwardExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const ValuePtrList &input_values) const {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto &prim_name = primitive->name();

  if (input_values.size() < kSize3) {
    MS_EXCEPTION(ValueError) << "For gradient of L1Loss, the input size is illegal.";
  }
  auto temp_shapes = {input_values[kInputIndex1], input_values[kInputIndex2]};
  return {BroadCastInferShape(prim_name, temp_shapes)};
}

REGISTER_SIMPLE_INFER(kNameL1LossBackwardExt, L1LossBackwardExtFuncImpl)
}  // namespace ops
}  // namespace mindspore
