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

#include "infer/ops_func_impl/add_rmsnorm_dynamic_quant.h"
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
abstract::BaseShapePtr AddRmsNormDynamicQuantFuncImpl::InferShape(
  const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const {
  auto x1_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto gamma_shape_ptr = input_args[kInputIndex2]->GetShape();

  auto x1_shape = x1_shape_ptr->GetShapeVector();
  auto x2_shape = input_args[kInputIndex1]->GetShape()->GetShapeVector();
  auto gamma_shape = gamma_shape_ptr->GetShapeVector();

  const auto &smooth_scale1 = input_args[kInputIndex3];
  const auto &smooth_scale2 = input_args[kInputIndex4];

  auto is_x1_static = !IsDynamic(x1_shape);
  auto is_x2_static = !IsDynamic(x2_shape);
  if (smooth_scale1->GetType()->type_id() != kMetaTypeNone) {
    auto smooth_scale1_shape = smooth_scale1->GetShape()->GetShapeVector();
    MS_CHECK_VALUE(
      smooth_scale1_shape.size() == 1,
      CheckAndConvertUtils::FormatCommMsg("The rank of smooth_scale1 must be 1, but got: ", smooth_scale1_shape));

    if (!IsDynamic(smooth_scale1_shape) && is_x1_static) {
      MS_CHECK_VALUE(smooth_scale1_shape[0] == x1_shape[x1_shape.size() - 1],
                     CheckAndConvertUtils::FormatCommMsg(
                       "The dim of smooth_scale1 must equal to the last dim of x_shape, but got smooth_scale1_shape: ",
                       smooth_scale1_shape, ", x_shape: ", x1_shape));
    }
  }

  if (smooth_scale2->GetType()->type_id() != kMetaTypeNone) {
    auto smooth_scale2_shape = smooth_scale2->GetShape()->GetShapeVector();
    MS_CHECK_VALUE(
      smooth_scale2_shape.size() == 1,
      CheckAndConvertUtils::FormatCommMsg("The rank of smooth_scale1 must be 1, but got: ", smooth_scale2_shape));

    if (!IsDynamic(smooth_scale2_shape) && is_x1_static) {
      MS_CHECK_VALUE(smooth_scale2_shape[0] == x1_shape[x1_shape.size() - 1],
                     CheckAndConvertUtils::FormatCommMsg(
                       "The dim of smooth_scale2 must equal to the last dim of x_shape, but got smooth_scale2_shape: ",
                       smooth_scale2_shape, ", x_shape: ", x1_shape));
    }

    if (smooth_scale1->GetType()->type_id() == kMetaTypeNone) {
      MS_EXCEPTION(TypeError) << "smooth_scale1 can't be None when smooth_scale2 is not None";
    }
  }

  MS_CHECK_VALUE(gamma_shape.size() == 1,
                 CheckAndConvertUtils::FormatCommMsg("The rank of gamma must be 1, but got: ", gamma_shape));
  if (!IsDynamic(gamma_shape) && is_x1_static) {
    MS_CHECK_VALUE(gamma_shape[0] == x1_shape[x1_shape.size() - 1],
                   CheckAndConvertUtils::FormatCommMsg(
                     "The dim of gamma_shape must equal to the last dim of x_shape, but got gamma_shape: ", gamma_shape,
                     ", x_shape: ", x1_shape));
  }

  BaseShapePtr out_scale_shape_ptr;
  if (IsDynamicRank(x1_shape)) {
    out_scale_shape_ptr = x1_shape_ptr;
  } else {
    MS_CHECK_VALUE(x1_shape.size() > 1, CheckAndConvertUtils::FormatCommMsg(
                                          "The rank of input x must be greater than 1, but got: ", x1_shape));
    out_scale_shape_ptr = std::make_shared<abstract::Shape>(ShapeVector(x1_shape.begin(), x1_shape.end() - 1));
  }

  if (is_x1_static & is_x2_static) {
    MS_CHECK_VALUE(x1_shape == x2_shape,
                   CheckAndConvertUtils::FormatCommMsg(
                     "The shape of x1 and x2 must be equal, but got x1_shape: ", x1_shape, ", x2_shape: ", x2_shape));
  }

  std::vector<BaseShapePtr> shapes_list = {x1_shape_ptr, x1_shape_ptr, x1_shape_ptr, out_scale_shape_ptr,
                                           out_scale_shape_ptr};
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

ShapeArray AddRmsNormDynamicQuantFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const ValuePtrList &input_values) const {
  const auto smooth_scale1 = input_values[kInputIndex3];
  const auto smooth_scale2 = input_values[kInputIndex4];
  if (smooth_scale1 != mindspore::kNone) {
    const auto &smooth_scale1_tensor = smooth_scale1->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(smooth_scale1_tensor);
    const auto &smooth_scale1_shape = smooth_scale1_tensor->shape();
    MS_CHECK_VALUE(
      smooth_scale1_shape.size() == 1,
      CheckAndConvertUtils::FormatCommMsg("The rank of smooth_scale1 must be 1, but got: ", smooth_scale1_shape));
  }

  if (smooth_scale2 != mindspore::kNone) {
    const auto &smooth_scale2_tensor = smooth_scale2->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(smooth_scale2_tensor);
    const auto &smooth_scale2_shape = smooth_scale2_tensor->shape();
    MS_CHECK_VALUE(
      smooth_scale2_shape.size() == 1,
      CheckAndConvertUtils::FormatCommMsg("The rank of smooth_scale1 must be 1, but got: ", smooth_scale2_shape));
    if (smooth_scale1 == mindspore::kNone) {
      MS_EXCEPTION(TypeError) << "smooth_scale1 can't be None when smooth_scale2 is not None";
    }
  }

  const auto &gamma_tensor = input_values[kInputIndex2]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(gamma_tensor);
  const auto &gamma_shape = gamma_tensor->shape();
  MS_CHECK_VALUE(gamma_shape.size() == 1,
                 CheckAndConvertUtils::FormatCommMsg("The rank of gamma must be 1, but got: ", gamma_shape));

  const auto &x1_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x1_tensor);
  const auto &x2_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x2_tensor);
  const auto &x1_shape = x1_tensor->shape();
  const auto &x2_shape = x2_tensor->shape();

  MS_CHECK_VALUE(x1_shape.size() > 1, CheckAndConvertUtils::FormatCommMsg(
                                        "The rank of input x must be greater than 1, but got: ", x1_shape));
  MS_CHECK_VALUE(x1_shape == x2_shape,
                 CheckAndConvertUtils::FormatCommMsg(
                   "The shape of x1 and x2 must be equal, but got x1_shape: ", x1_shape, ", x2_shape: ", x2_shape));

  ShapeVector out_scale_shape(x1_shape.begin(), x1_shape.end() - 1);
  return {x1_shape, x1_shape, x1_shape, out_scale_shape, out_scale_shape};
}

TypePtr AddRmsNormDynamicQuantFuncImpl::InferType(const PrimitivePtr &prim,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  auto x1_type = input_args[kInputIndex0]->GetType();
  auto x2_type = input_args[kInputIndex1]->GetType();
  auto gamma_type = input_args[kInputIndex2]->GetType();
  auto smooth_scale1_type = input_args[kInputIndex3]->GetType();
  auto smooth_scale2_type = input_args[kInputIndex4]->GetType();

  auto quant_out_type = std::make_shared<TensorType>(kInt8);
  auto scale_out_type = std::make_shared<TensorType>(kFloat32);

  std::map<std::string, TypePtr> types = {{"x1_type", x1_type},
                                          {"x2_type", x2_type},
                                          {"gamma_type", gamma_type},
                                          {"smooth_scale1_type", smooth_scale1_type},
                                          {"smooth_scale2_type", smooth_scale2_type}};
  const std::set<TypePtr> valid_types = {kFloat16, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  std::vector<TypePtr> types_list = {quant_out_type, quant_out_type, x1_type, scale_out_type, scale_out_type};
  return std::make_shared<Tuple>(types_list);
}

TypePtrList AddRmsNormDynamicQuantFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const ValuePtrList &input_values) const {
  const auto &x1_tensor = input_values[kInputIndex0]->cast<tensor::TensorPtr>();
  const auto &x2_tensor = input_values[kInputIndex1]->cast<tensor::TensorPtr>();
  const auto &gamma_tensor = input_values[kInputIndex2]->cast<tensor::TensorPtr>();
  const auto &smooth_scale1_tensor = input_values[kInputIndex3]->cast<tensor::TensorPtr>();
  const auto &smooth_scale2_tensor = input_values[kInputIndex4]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x1_tensor);
  MS_EXCEPTION_IF_NULL(x2_tensor);
  MS_EXCEPTION_IF_NULL(gamma_tensor);
  MS_EXCEPTION_IF_NULL(smooth_scale1_tensor);
  MS_EXCEPTION_IF_NULL(smooth_scale2_tensor);

  auto x1_type = x1_tensor->Dtype();
  auto x2_type = x2_tensor->Dtype();
  auto gamma_type = gamma_tensor->Dtype();
  auto smooth_scale1_type = smooth_scale1_tensor->Dtype();
  auto smooth_scale2_type = smooth_scale2_tensor->Dtype();

  auto quant_out_type = std::make_shared<TensorType>(kInt8);
  auto scale_out_type = std::make_shared<TensorType>(kFloat32);

  std::map<std::string, TypePtr> types = {{"x1_type", x1_type},
                                          {"x2_type", x2_type},
                                          {"gamma_type", gamma_type},
                                          {"smooth_scale1_type", smooth_scale1_type},
                                          {"smooth_scale2_type", smooth_scale2_type}};
  const std::set<TypePtr> valid_types = {kFloat16, kBFloat16};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());

  return {quant_out_type, quant_out_type, x1_type, scale_out_type, scale_out_type};
}

}  // namespace ops
}  // namespace mindspore
