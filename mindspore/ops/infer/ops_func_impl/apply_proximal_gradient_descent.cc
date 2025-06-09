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

#include "infer/ops_func_impl/apply_proximal_gradient_descent.h"

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
abstract::ShapePtr ApplyProximalGradientDescentInferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(var_shape_ptr)[kShape];
  auto alpha_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
  auto l1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShape())[kShape];
  auto l2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->GetShape())[kShape];
  auto delta_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->GetShape())[kShape];
  // dynamic rank
  if (IsDynamicRank(var_shape) || IsDynamicRank(delta_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }
  (void)CheckAndConvertUtils::CheckInteger("alpha_shape size", SizeToLong(alpha_shape.size()), kLessEqual, batch_rank,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("l1_shape size", SizeToLong(l1_shape.size()), kLessEqual, batch_rank,
                                           prim_name);
  (void)CheckAndConvertUtils::CheckInteger("l2_shape size", SizeToLong(l2_shape.size()), kLessEqual, batch_rank,
                                           prim_name);
  ShapeVector output_vec;
  for (size_t i = 0; i < var_shape.size(); i++) {
    output_vec.push_back(var_shape[i]);
  }
  abstract::ShapePtr output_shape_ptr = std::make_shared<abstract::Shape>(output_vec);
  // dynamic shape
  if (IsDynamic(var_shape) || IsDynamic(delta_shape)) {
    return output_shape_ptr;
  }
  // var and delta must have the same shape
  if (var_shape != delta_shape) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', evaluator arg 'delta' must have the same shape as 'var'. But got 'delta' shape: "
                             << delta_shape << ", 'var' shape: " << var_shape << ".";
  }
  return output_shape_ptr;
}

TypePtr ApplyProximalGradientDescentInferType(const PrimitivePtr &prim,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto var_type = input_args[kInputIndex0]->GetType();
  auto alpha_type = input_args[kInputIndex1]->GetType();
  auto l1_type = input_args[kInputIndex2]->GetType();
  auto l2_type = input_args[kInputIndex3]->GetType();
  auto delta_type = input_args[kInputIndex4]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  // var, delta must have the same type as var
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var_type", var_type));
  (void)args.insert(std::make_pair("delta_type", delta_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // alpha、l1、l2 must be a scalar type
  std::map<std::string, TypePtr> args_alpha;
  std::map<std::string, TypePtr> args_l1;
  std::map<std::string, TypePtr> args_l2;
  (void)args_alpha.insert(std::make_pair("alpha_type", alpha_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_alpha, valid_types, prim_name);
  (void)args_l1.insert(std::make_pair("l1_type", l1_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l1, valid_types, prim_name);
  (void)args_l2.insert(std::make_pair("l2_type", l2_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_l2, valid_types, prim_name);
  return var_type;
}
}  // namespace

BaseShapePtr ApplyProximalGradientDescentFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                              const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyProximalGradientDescentInferShape(primitive, input_args);
}

TypePtr ApplyProximalGradientDescentFuncImpl::InferType(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyProximalGradientDescentInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
