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

#include "infer/ops_func_impl/apply_adagrad_v2.h"

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
BaseShapePtr ApplyAdagradV2InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto var_shape = input_args[kInputIndex0]->GetShape();
  auto accum_shape = input_args[kInputIndex1]->GetShape();
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShape())[kShape];
  auto grad_shape = input_args[kInputIndex3]->GetShape();
  auto grad_shape_ptr = grad_shape->cast<abstract::ShapePtr>();
  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    batch_rank = GetValue<int64_t>(primitive->GetAttr(kBatchRank));
  }
  // lr must be a scalar [Number, Tensor]
  const int64_t kShapeSize_ = 1;
  auto lr_shape_size = lr_shape.size();
  if (batch_rank > 0) {
    // when batch dimension exists, the rank of `lr` must equal to batch_rank.
    (void)CheckAndConvertUtils::CheckInteger("lr's rank'", lr_shape_size, kEqual, batch_rank, primitive->name());
  } else {
    (void)CheckAndConvertUtils::CheckInteger("lr's rank'", SizeToLong(lr_shape_size), kLessEqual, kShapeSize_,
                                             primitive->name());
  }
  // var, accum and grad must have the same shape
  MS_EXCEPTION_IF_NULL(grad_shape_ptr);
  if (grad_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape, accum_shape});
  }
  std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
  (void)same_shape_args_map.emplace("accum", accum_shape);
  (void)same_shape_args_map.emplace("grad", grad_shape);
  for (auto &elem : same_shape_args_map) {
    if (*elem.second != *var_shape) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', evaluator arg '" << elem.first
                               << "' and 'var' must have the same shape. But got '" << elem.first
                               << "' shape: " << elem.second->ToString() << ", 'var' shape: " << var_shape->ToString()
                               << ".";
    }
  }
  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{var_shape, accum_shape});
}

TypePtr ApplyAdagradV2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto var_type = input_args[kInputIndex0]->GetType();
  auto accum_type = input_args[kInputIndex1]->GetType();
  auto lr_type = input_args[kInputIndex2]->GetType();
  auto grad_type = input_args[kInputIndex3]->GetType();
  const std::set<TypePtr> valid_types = {kFloat};
  // var, accum, grad  must have the same type
  std::map<std::string, TypePtr> args;
  (void)args.emplace("var_type", var_type);
  (void)args.emplace("accum_type", accum_type);
  (void)args.emplace("grad_type", grad_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, valid_types, primitive->name());
  // lr mustr be a scalar
  std::map<std::string, TypePtr> args_lr;
  (void)args_lr.emplace("lr_type", lr_type);
  (void)CheckAndConvertUtils::CheckScalarOrTensorTypesSame(args_lr, valid_types, primitive->name());
  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, accum_type});
}
}  // namespace

BaseShapePtr ApplyAdagradV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyAdagradV2InferShape(primitive, input_args);
}

TypePtr ApplyAdagradV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyAdagradV2InferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore