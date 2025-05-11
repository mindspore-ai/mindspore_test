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

#include "infer/ops_func_impl/sparse_apply_proximal_gradient_descent.h"

#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "base/float16.h"
#include "ir/anf.h"
#include "ir/primitive.h"
#include "ir/tensor.h"
#include "mindapi/base/type_id.h"
#include "mindapi/helper.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr SparseApplyProximalGradientDescentInferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape())[kShape];
  auto alpha_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShape())[kShape];
  auto l1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->GetShape())[kShape];
  auto l2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->GetShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->GetShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->GetShape())[kShape];

  std::vector<ShapeVector> scalar_shapes = {alpha_shape, l1_shape, l2_shape};
  auto is_dynamic_scalar = std::any_of(scalar_shapes.begin(), scalar_shapes.end(), IsDynamic);
  if (!is_dynamic_scalar) {
    int64_t scalar_shape = 0;
    (void)CheckAndConvertUtils::CheckInteger("alpha_shape size", SizeToLong(alpha_shape.size()), kEqual, scalar_shape,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("l1_shape size", SizeToLong(l1_shape.size()), kEqual, scalar_shape,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("l2_shape size", SizeToLong(l2_shape.size()), kEqual, scalar_shape,
                                             prim_name);
  }

  // Var dimension must be equal or greater than 1.
  (void)CheckAndConvertUtils::CheckInteger("var dimension", SizeToLong(var_shape.size()), kGreaterEqual, 1, prim_name);
  // Indices must be rank 1.
  (void)CheckAndConvertUtils::CheckInteger("indices dimension", SizeToLong(indices_shape.size()), kEqual, 1, prim_name);
  auto is_dynamic = IsDynamic(var_shape) || IsDynamic(grad_shape) || IsDynamic(indices_shape);
  if (!is_dynamic) {
    if (var_shape.size() != grad_shape.size()) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', rank(grad) should be same as rank(var), but got rank(grad): " << grad_shape.size()
                               << ", rank(var): " << var_shape.size() << ".";
    }
    for (size_t i = 1; i < var_shape.size(); ++i) {
      if (var_shape[i] != grad_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "'. the shape of var and grad must equal in dimension " << i
                                 << ".";
      }
    }
    if (indices_shape[0] != grad_shape[0]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', grad.shape[0] must be equal to indices.shape[0], but got grad.shape[0]: "
                               << grad_shape[0] << ", indices.shape[0]: " << indices_shape[0] << ".";
    }
  }

  return std::make_shared<abstract::Shape>(var_shape);
}

TypePtr SparseApplyProximalGradientDescentInferType(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_type = input_args[0]->GetType();
  auto alpha_type = input_args[1]->GetType();
  auto l1_type = input_args[2]->GetType();
  auto l2_type = input_args[3]->GetType();
  auto grad_type = input_args[4]->GetType();
  auto indices_type = input_args[5]->GetType();

  (void)CheckAndConvertUtils::CheckTensorTypeValid("var", var_type, common_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("alpha", alpha_type, common_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("l1", l1_type, common_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("l2", l2_type, common_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grad", grad_type, common_valid_types, prim_name);

  const std::set<TypePtr> valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, valid_types, prim_name);
  return var_type;
}
}  // namespace

BaseShapePtr SparseApplyProximalGradientDescentFuncImpl::InferShape(
  const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return SparseApplyProximalGradientDescentInferShape(primitive, input_args);
}

TypePtr SparseApplyProximalGradientDescentFuncImpl::InferType(const PrimitivePtr &primitive,
                                                              const std::vector<AbstractBasePtr> &input_args) const {
  return SparseApplyProximalGradientDescentInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
