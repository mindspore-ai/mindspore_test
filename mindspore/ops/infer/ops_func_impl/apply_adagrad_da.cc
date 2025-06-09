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

#include "infer/ops_func_impl/apply_adagrad_da.h"

#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <utility>

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
BaseShapePtr ApplyAdagradDAInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
  auto gradient_accumulator_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
  auto gradient_squared_accumulator_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->GetShape())[kShape];
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, gradient_accumulator_shape, prim_name);
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, gradient_squared_accumulator_shape, prim_name);
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, grad_shape, prim_name);

  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->GetShape())[kShape];
  auto l1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->GetShape())[kShape];
  auto l2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex6]->GetShape())[kShape];
  auto global_step_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex7]->GetShape())[kShape];
  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }
  if (batch_rank < 0) {
    MS_EXCEPTION(ValueError) << "batch_rank is" << batch_rank;
  }
  auto lr_shape_rank = SizeToLong(lr_shape.size());
  auto l1_shape_rank = SizeToLong(l1_shape.size());
  auto l2_shape_rank = SizeToLong(l2_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", lr_shape_rank, kEqual, batch_rank, primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("l1_shape size", l1_shape_rank, kEqual, batch_rank, primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("l2_shape size", l2_shape_rank, kEqual, batch_rank, primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("global_step_shape size", global_step_shape.size(), kEqual, batch_rank,
                                           primitive->name());
  return var_shape_ptr;
}

TypePtr ApplyAdagradDAInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto var_type = input_args[kInputIndex0]->GetType();
  auto grad_type = input_args[kInputIndex3]->GetType();
  auto lr_type = input_args[kInputIndex4]->GetType();
  auto l1_type = input_args[kInputIndex5]->GetType();
  auto l2_type = input_args[kInputIndex6]->GetType();
  auto global_step_type = input_args[kInputIndex7]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // gradient_accumulator、gradient_squared_accumulator、grad must have the same type as var
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var_type", var_type));
  (void)args.insert(std::make_pair("grad_type", grad_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // lr、l1、l2、global_step_type must be a scalar type
  std::map<std::string, TypePtr> args_lr;
  std::map<std::string, TypePtr> args_l1;
  std::map<std::string, TypePtr> args_l2;
  std::map<std::string, TypePtr> args_global_step;
  (void)args_lr.insert(std::make_pair("lr_type", lr_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, prim_name);
  (void)args_l1.insert(std::make_pair("l1_type", l1_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l1, valid_types, prim_name);
  (void)args_l2.insert(std::make_pair("l2_type", l2_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l2, valid_types, prim_name);
  (void)args_global_step.insert(std::make_pair("global_step_type", global_step_type));
  const std::set<TypePtr> valid_types1 = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_global_step, valid_types1, prim_name);
  return var_type;
}
}  // namespace

BaseShapePtr ApplyAdagradDAFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyAdagradDAInferShape(primitive, input_args);
}

TypePtr ApplyAdagradDAFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyAdagradDAInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
