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

#include "infer/ops_func_impl/rmsnorm_quant.h"
#include <vector>
#include <memory>
#include <map>
#include <string>
#include <set>
#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
abstract::BaseShapePtr RmsNormQuantFuncImpl::InferShape(const PrimitivePtr &prim,
                                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto gamma_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto beta_shape_ptr = input_args[kInputIndex2]->GetShape();

  auto x_shape = x_shape_ptr->GetShapeVector();
  auto gamma_shape = gamma_shape_ptr->GetShapeVector();
  auto beta_shape = beta_shape_ptr->GetShapeVector();

  if (!IsDynamic(gamma_shape) && !IsDynamic(x_shape) && !IsDynamic(beta_shape)) {
    MS_CHECK_VALUE(gamma_shape[gamma_shape.size() - 1] == x_shape[x_shape.size() - 1],
                   CheckAndConvertUtils::FormatCommMsg(
                     "The dim of gamma_shape must equal to the last dim of x_shape, but got gamma_shape: ", gamma_shape,
                     ", x_shape: ", x_shape));

    MS_CHECK_VALUE(beta_shape[beta_shape.size() - 1] == x_shape[x_shape.size() - 1],
                   CheckAndConvertUtils::FormatCommMsg(
                     "The dim of beta_shape must equal to the last dim of x_shape, but got beta_shape: ", beta_shape,
                     ", x_shape: ", x_shape));
  }

  std::vector<BaseShapePtr> shapes_list = {x_shape_ptr};
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

ShapeArray RmsNormQuantFuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &gamma_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(gamma_tensor);
  const auto &gamma_shape = gamma_tensor->shape();

  const auto &beta_tensor = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(beta_tensor);
  const auto &beta_shape = beta_tensor->shape();

  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_shape = x_tensor->shape();

  MS_CHECK_VALUE(gamma_shape[gamma_shape.size() - 1] == x_shape[x_shape.size() - 1],
                 CheckAndConvertUtils::FormatCommMsg(
                   "The dim of gamma_shape must equal to the last dim of x_shape, but got gamma_shape: ", gamma_shape,
                   ", x_shape: ", x_shape));

  MS_CHECK_VALUE(beta_shape[beta_shape.size() - 1] == x_shape[x_shape.size() - 1],
                 CheckAndConvertUtils::FormatCommMsg(
                   "The dim of beta_shape must equal to the last dim of x_shape, but got beta_shape: ", beta_shape,
                   ", x_shape: ", x_shape));

  return {x_shape};
}

TypePtr RmsNormQuantFuncImpl::InferType(const PrimitivePtr &prim,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto x_type = input_args[kInputIndex0]->GetType();
  auto gamma_type = input_args[kInputIndex1]->GetType();
  auto beta_type = input_args[kInputIndex2]->GetType();
  auto scale_type = input_args[kInputIndex3]->GetType();
  auto offset_type = input_args[kInputIndex4]->GetType();

  auto quant_out_type = std::make_shared<TensorType>(kInt8);

  std::map<std::string, TypePtr> types = {
    {"x_type", x_type}, {"gamma_type", gamma_type}, {"beta_type", beta_type}, {"scale_type", scale_type}};
  const std::set<TypePtr> valid_types = {kFloat16, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());

  CheckAndConvertUtils::CheckTensorTypeValid("offset_type", offset_type, {kInt8}, prim->name());

  std::vector<TypePtr> types_list = {quant_out_type};
  return std::make_shared<Tuple>(types_list);
}

TypePtrList RmsNormQuantFuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  const auto &gamma_tensor = input_values[kInputIndex1]->cast<tensor::BaseTensorPtr>();
  const auto &beta_tensor = input_values[kInputIndex2]->cast<tensor::BaseTensorPtr>();
  const auto &scale_tensor = input_values[kInputIndex3]->cast<tensor::BaseTensorPtr>();
  const auto &offset_tensor = input_values[kInputIndex4]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  MS_EXCEPTION_IF_NULL(gamma_tensor);
  MS_EXCEPTION_IF_NULL(beta_tensor);
  MS_EXCEPTION_IF_NULL(scale_tensor);
  MS_EXCEPTION_IF_NULL(offset_tensor);

  auto x_type = x_tensor->Dtype();
  auto gamma_type = gamma_tensor->Dtype();
  auto beta_type = beta_tensor->Dtype();
  auto scale_type = scale_tensor->Dtype();
  auto offset_type = offset_tensor->Dtype();

  auto quant_out_type = std::make_shared<TensorType>(kInt8);

  std::map<std::string, TypePtr> types = {
    {"x_type", x_type}, {"gamma_type", gamma_type}, {"beta_type", beta_type}, {"scale_type", scale_type}};
  const std::set<TypePtr> valid_types = {kFloat16, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());

  CheckAndConvertUtils::CheckTensorTypeValid("offset_type", offset_type, {kInt8}, primitive->name());

  return {quant_out_type};
}

}  // namespace ops
}  // namespace mindspore
