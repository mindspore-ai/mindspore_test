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

#include "infer/ops_func_impl/apply_proximal_adagrad.h"

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
BaseShapePtr ApplyProximalAdagradInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto var_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto accum_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShape())[kShape];
  auto l1_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->GetShape())[kShape];
  auto l2_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->GetShape())[kShape];
  auto grad_shape_ptr = input_args[kInputIndex5]->GetShape();

  size_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }

  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", SizeToLong(lr_shape.size()), kEqual, batch_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("l1_shape size", SizeToLong(l1_shape.size()), kEqual, batch_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("l2_shape size", SizeToLong(l2_shape.size()), kEqual, batch_rank, prim_name);

  if (grad_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr});
  }

  // var, accum and grad must have the same shape
  std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
  (void)same_shape_args_map.insert(std::make_pair("accum", accum_shape_ptr));
  (void)same_shape_args_map.insert(std::make_pair("grad", grad_shape_ptr));
  for (auto &elem : same_shape_args_map) {
    if (*elem.second != *var_shape_ptr) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', evaluator arg '" << elem.first
                               << "' must have the same shape as 'var'. But got '" << elem.first
                               << "' shape: " << elem.second->ToString()
                               << ", 'var' shape: " << var_shape_ptr->ToString() << ".";
    }
  }

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape_ptr, accum_shape_ptr});
}

TypePtr ApplyProximalAdagradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto var_type = input_args[kInputIndex0]->GetType();
  auto accum_type = input_args[kInputIndex1]->GetType();
  auto lr_type = input_args[kInputIndex2]->GetType();
  auto l1_type = input_args[kInputIndex3]->GetType();
  auto l2_type = input_args[kInputIndex4]->GetType();
  auto grad_type = input_args[kInputIndex5]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  // var, accum and grad must have the same type
  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("var", var_type));
  (void)args.insert(std::make_pair("accum", accum_type));
  (void)args.insert(std::make_pair("grad", grad_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, prim_name);
  // lr, l1, l2 type must be valid
  std::map<std::string, TypePtr> args_lr;
  (void)args_lr.insert(std::make_pair("lr", lr_type));
  std::map<std::string, TypePtr> args_l1;
  (void)args_l1.insert(std::make_pair("l1", l1_type));
  std::map<std::string, TypePtr> args_l2;
  (void)args_l2.insert(std::make_pair("l2", l2_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args_lr, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args_l1, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args_l2, valid_types, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type});
}
}  // namespace

BaseShapePtr ApplyProximalAdagradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyProximalAdagradInferShape(primitive, input_args);
}

TypePtr ApplyProximalAdagradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyProximalAdagradInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
