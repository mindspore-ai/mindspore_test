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

#include "infer/ops_func_impl/apply_keras_momentum.h"

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
BaseShapePtr ApplyKerasMomentumInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto var_shape = input_args[0]->GetShape();
  auto accum_shape = input_args[1]->GetShape();
  auto grad_shape = input_args[3]->GetShape();

  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[2]->GetShapeTrack())[kShape];
  auto momentum_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->GetShapeTrack())[kShape];
  auto momentum_shape_rank = SizeToLong(momentum_shape.size());

  // lr, momentum must be scalar
  if (!IsDynamic(lr_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("lr_shape rank", SizeToLong(lr_shape.size()), kEqual, 0, prim_name);
  }
  if (!IsDynamic(momentum_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("momentum_shape rank", momentum_shape_rank, kEqual, 0, prim_name);
  }

  // var, accum and grad must have the same shape
  std::vector<abstract::BaseShapePtr> check_shapes = {var_shape, accum_shape, grad_shape};
  auto is_dynamic = std::any_of(check_shapes.begin(), check_shapes.end(),
                                [&](const abstract::BaseShapePtr &shape) { return shape->IsDynamic(); });
  if (!is_dynamic) {
    std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
    (void)same_shape_args_map.insert(std::make_pair("accum", accum_shape));
    (void)same_shape_args_map.insert(std::make_pair("grad", grad_shape));
    for (auto &elem : same_shape_args_map) {
      if (*elem.second != *var_shape) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', evaluator arg '" << elem.first
                                 << "' must have the same shape as 'var'. But got '" << elem.first
                                 << "' shape: " << elem.second->ToString() << ", 'var' shape: " << var_shape->ToString()
                                 << ".";
      }
    }
  }

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape, accum_shape});
}

TypePtr ApplyKerasMomentumInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();

  auto var_type = input_args[0]->GetType();
  auto accum_type = input_args[1]->GetType();
  auto lr_type = input_args[2]->GetType();
  auto grad_type = input_args[3]->GetType();
  auto momentum_type = input_args[4]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // var, accum and grad must have the same type
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var", var_type));
  (void)args.insert(std::make_pair("accum", accum_type));
  (void)args.insert(std::make_pair("grad", grad_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // lr, momentum type must be valid
  (void)CheckAndConvertUtils::CheckTensorTypeValid("lr_dtype", lr_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("momentum_dtype", momentum_type, valid_types, prim_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type});
}
}  // namespace
BaseShapePtr ApplyKerasMomentumFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyKerasMomentumInferShape(primitive, input_args);
}

TypePtr ApplyKerasMomentumFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyKerasMomentumInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
