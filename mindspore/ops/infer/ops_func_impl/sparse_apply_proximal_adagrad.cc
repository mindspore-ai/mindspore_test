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

#include "infer/ops_func_impl/sparse_apply_proximal_adagrad.h"

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
void SparseApplyProximalAdagradCheckTensorShapeAndSize(const ShapeVector &var_shape, const ShapeVector &accum_shape,
                                                       const ShapeVector &grad_shape, const ShapeVector &indices_shape,
                                                       const std::string &prim_name) {
  std::vector<ShapeVector> check_shapes = {var_shape, accum_shape, grad_shape, indices_shape};
  auto is_dynamic = std::any_of(check_shapes.begin(), check_shapes.end(), IsDynamic);
  auto is_dynamic_rank = std::any_of(check_shapes.begin(), check_shapes.end(), IsDynamicRank);
  // Var dimension must be equal or greater than 1.
  (void)CheckAndConvertUtils::CheckInteger("var dimension", SizeToLong(var_shape.size()), kGreaterEqual, 1, prim_name);
  // Indices must be rank 1.
  (void)CheckAndConvertUtils::CheckInteger("indices dimension", SizeToLong(indices_shape.size()), kEqual, 1, prim_name);

  if (!is_dynamic_rank) {
    if (var_shape.size() != accum_shape.size()) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', rank(accum) should be same as rank(var), but got rank(grad): "
                               << accum_shape.size() << ", rank(var): " << var_shape.size() << ".";
    }
    if (var_shape.size() != grad_shape.size()) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', rank(grad) should be same as rank(var), but got rank(grad): " << grad_shape.size()
                               << ", rank(var): " << var_shape.size() << ".";
    }
  }

  if (!is_dynamic) {
    if (indices_shape[0] != grad_shape[0]) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', grad.shape[0] must be equal to indices.shape[0], but got grad.shape[0]: "
                               << grad_shape[0] << ", indices.shape[0]: " << indices_shape[0] << ".";
    }
    const size_t kZeroNum = 0;
    for (size_t i = 0; i < var_shape.size(); ++i) {
      if (var_shape[i] != accum_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "'. the shape of var and accum must equal in dimension "
                                 << i << ".";
      }
      if (i == kZeroNum) {
        continue;
      }
      if (var_shape[i] != grad_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "'. the shape of var and grad must equal in dimension " << i
                                 << ".";
      }
    }
  }
}

BaseShapePtr SparseApplyProximalAdagradInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShape())[kShape];
  auto accum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->GetShape())[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->GetShape())[kShape];
  auto l1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->GetShape())[kShape];
  auto l2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->GetShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->GetShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[6]->GetShape())[kShape];

  SparseApplyProximalAdagradCheckTensorShapeAndSize(var_shape, accum_shape, grad_shape, indices_shape, prim_name);

  if (!(IsDynamic(lr_shape) || IsDynamic(l1_shape) || IsDynamic(l2_shape))) {
    const int64_t scalar_shape = 0;
    (void)CheckAndConvertUtils::CheckInteger("lr_shape size", SizeToLong(lr_shape.size()), kEqual, scalar_shape,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("l1_shape size", SizeToLong(l1_shape.size()), kEqual, scalar_shape,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckInteger("l2_shape size", SizeToLong(l2_shape.size()), kEqual, scalar_shape,
                                             prim_name);
  }

  abstract::ShapePtr var_shape_ptr = std::make_shared<abstract::Shape>(var_shape);
  abstract::ShapePtr accum_shape_ptr = std::make_shared<abstract::Shape>(accum_shape);
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr});
}

TypePtr SparseApplyProximalAdagradInferType(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_type = input_args[0]->GetType();
  auto accum_type = input_args[1]->GetType();
  auto lr_type = input_args[2]->GetType();
  auto l1_type = input_args[3]->GetType();
  auto l2_type = input_args[4]->GetType();
  auto grad_type = input_args[5]->GetType();
  auto indices_type = input_args[6]->GetType();

  std::set<TypePtr> tensor_valid_types = {kFloat16, kFloat32};
  std::map<std::string, TypePtr> args;
  (void)args.insert({"var", var_type});
  (void)args.insert({"accum", accum_type});
  (void)args.insert({"grad", grad_type});
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, tensor_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame({{"lr", lr_type}}, tensor_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame({{"l1", l1_type}}, tensor_valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame({{"l2", l2_type}}, tensor_valid_types, prim_name);

  const std::set<TypePtr> valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, valid_types, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type});
}
}  // namespace

BaseShapePtr SparseApplyProximalAdagradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                            const std::vector<AbstractBasePtr> &input_args) const {
  return SparseApplyProximalAdagradInferShape(primitive, input_args);
}

TypePtr SparseApplyProximalAdagradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) const {
  return SparseApplyProximalAdagradInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
