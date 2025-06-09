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

#include "infer/ops_func_impl/apply_gradient_descent.h"

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
BaseShapePtr ApplyGradientDescentInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto var_shape = input_args[kInputIndex0]->GetShape();
  if (IsDynamicRank(CheckAndConvertUtils::ConvertShapePtrToShapeMap(var_shape)[kShape])) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto alpha_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
  if (IsDynamicRank(alpha_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  auto delta_shape = input_args[kInputIndex2]->GetShape();
  if (IsDynamicRank(CheckAndConvertUtils::ConvertShapePtrToShapeMap(delta_shape)[kShape])) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  // var and delta must have the same shape when is not dynamic
  auto var_shape_ptr = var_shape->cast<abstract::ShapePtr>();
  auto delta_shape_ptr = delta_shape->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(var_shape_ptr);
  if (!var_shape_ptr->IsDynamic() && !delta_shape_ptr->IsDynamic()) {
    if (*var_shape != *delta_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name
                               << "', evaluator arg 'delta' must have the same shape as 'var'. But got 'delta' shape: "
                               << delta_shape->ToString() << ", 'var' shape: " << var_shape->ToString() << ".";
    }
  }
  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    batch_rank = GetValue<int64_t>(primitive->GetAttr(kBatchRank));
  }
  // alpha must be a scalar [Number, Tensor]
  const int64_t kShapeSize = 1;
  auto alpha_shape_rank = SizeToLong(alpha_shape.size());
  if (batch_rank > 0) {
    // when batch dimension exists, the rank of `alpha` must equal to batch_rank.
    (void)CheckAndConvertUtils::CheckInteger("alpha's rank'", alpha_shape_rank, kEqual, batch_rank, prim_name);
  } else {
    (void)CheckAndConvertUtils::CheckInteger("alpha's rank'", alpha_shape_rank, kLessEqual, kShapeSize, prim_name);
    if (alpha_shape_rank == 1) {
      (void)CheckAndConvertUtils::CheckInteger("alpha_shape[0]", alpha_shape[0], kEqual, kShapeSize, primitive->name());
    }
  }
  return var_shape_ptr;
}

TypePtr ApplyGradientDescentInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto var_type = input_args[kInputIndex0]->GetType();
  auto alpha_type = input_args[kInputIndex1]->GetType();
  auto delta_type = input_args[kInputIndex2]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kInt8,   kUInt8,   kInt16,     kUInt16,    kInt32,
                                         kUInt32,  kInt64,   kUInt64, kFloat64, kComplex64, kComplex128};
  // delta must have the same type as var
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var_type", var_type));
  (void)args.insert(std::make_pair("delta_type", delta_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // alpha must be a scalar type
  std::map<std::string, TypePtr> args_alpha;
  (void)args_alpha.insert(std::make_pair("alpha_type", alpha_type));
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_alpha, valid_types, prim_name);
  return var_type;
}
}  // namespace

BaseShapePtr ApplyGradientDescentFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyGradientDescentInferShape(primitive, input_args);
}

TypePtr ApplyGradientDescentFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyGradientDescentInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
