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

#include "infer/ops_func_impl/apply_adam_with_amsgrad_v2.h"

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
BaseShapePtr ApplyAdamWithAmsgradV2InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto prim_name = primitive->name();
  auto var_shape = input_args[0]->GetShape();
  auto m_shape = input_args[1]->GetShape();
  auto v_shape = input_args[2]->GetShape();
  auto vhat_shape = input_args[3]->GetShape();
  auto beta1_power_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->GetShape())[kShape];
  auto beta2_power_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->GetShape())[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[6]->GetShape())[kShape];
  auto grad_shape = input_args[10]->GetShape();

  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }

  (void)CheckAndConvertUtils::CheckInteger("beta1_power_shape size", beta1_power_shape.size(), kGreaterEqual,
                                           batch_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("beta2_power_shape size", beta2_power_shape.size(), kGreaterEqual,
                                           batch_rank, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("lr_shape size", lr_shape.size(), kGreaterEqual, batch_rank, prim_name);

  if (var_shape->IsDynamic() || m_shape->IsDynamic() || v_shape->IsDynamic() || vhat_shape->IsDynamic() ||
      grad_shape->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{var_shape, m_shape, v_shape, vhat_shape});
  }

  // shape of var, m, v, vhat, grad must be the same
  std::map<std::string, abstract::BaseShapePtr> same_shape_args_map;
  (void)same_shape_args_map.insert(std::make_pair("m", m_shape));
  (void)same_shape_args_map.insert(std::make_pair("v", v_shape));
  (void)same_shape_args_map.insert(std::make_pair("vhat", vhat_shape));
  (void)same_shape_args_map.insert(std::make_pair("grad", grad_shape));
  for (auto &elem : same_shape_args_map) {
    if (*elem.second != *var_shape) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', evaluator arg '" << elem.first
                               << "' and 'var' must have the same shape. But got '" << elem.first
                               << "' shape: " << elem.second->ToString() << ", 'var' shape: " << var_shape->ToString()
                               << ".";
    }
  }
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape, m_shape, v_shape, vhat_shape});
}

TuplePtr ApplyAdamWithAmsgradV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  auto var_type = input_args[0]->GetType();
  auto m_type = input_args[1]->GetType();
  auto v_type = input_args[2]->GetType();
  auto vhat_type = input_args[3]->GetType();
  auto beta1_power_type = input_args[4]->GetType();
  auto beta2_power_type = input_args[5]->GetType();
  auto lr_type = input_args[6]->GetType();
  auto beta1_type = input_args[7]->GetType();
  auto beta2_type = input_args[8]->GetType();
  auto epsilon_type = input_args[9]->GetType();
  auto grad_type = input_args[10]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("var_type", var_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("m_type", m_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("v_type", v_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("vhat_type", vhat_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grad_type", grad_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("beta1_power_type", beta1_power_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("beta2_power_type", beta2_power_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("lr_type", lr_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("beta1_type", beta1_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("beta2_type", beta2_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("epsilon_type", epsilon_type, valid_types, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type, v_type, vhat_type});
}
}  // namespace

BaseShapePtr ApplyAdamWithAmsgradV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyAdamWithAmsgradV2InferShape(primitive, input_args);
}

TypePtr ApplyAdamWithAmsgradV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyAdamWithAmsgradV2InferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
