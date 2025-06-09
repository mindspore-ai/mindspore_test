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

#include "infer/ops_func_impl/fused_cast_adam_weight_decay.h"

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
BaseShapePtr FusedCastAdamWeightDecayFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                          const std::vector<AbstractBasePtr> &input_args) const {
  const auto &var_shape_ptr = input_args[kIndex0]->GetShape();
  const auto &var_shape = var_shape_ptr->GetShapeVector();

  const auto &m_shape_ptr = input_args[kIndex1]->GetShape();
  const auto &m_shape = var_shape_ptr->GetShapeVector();

  const auto &v_shape_ptr = input_args[kIndex2]->GetShape();
  const auto &v_shape = var_shape_ptr->GetShapeVector();

  const auto &grad_shape_ptr = input_args[kIndex8]->GetShape();
  const auto &grad_shape = var_shape_ptr->GetShapeVector();

  bool is_dynamic = IsDynamic(var_shape) || IsDynamic(m_shape) || IsDynamic(v_shape) || IsDynamic(grad_shape);
  if (!is_dynamic) {
    MS_CHECK_VALUE(var_shape == m_shape,
                   CheckAndConvertUtils::FormatCheckMsg("var_shape", var_shape, kEqual, m_shape, primitive));
    MS_CHECK_VALUE(var_shape == v_shape,
                   CheckAndConvertUtils::FormatCheckMsg("var_shape", var_shape, kEqual, v_shape, primitive));
    MS_CHECK_VALUE(var_shape == grad_shape,
                   CheckAndConvertUtils::FormatCheckMsg("var_shape", var_shape, kEqual, grad_shape, primitive));
  }

  return std::make_shared<abstract::TupleShape>(
    abstract::BaseShapePtrList({var_shape_ptr->Clone(), m_shape_ptr->Clone(), v_shape_ptr->Clone()}));
}

TypePtr FusedCastAdamWeightDecayFuncImpl::InferType(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = primitive->name();
  const auto &m_type = input_args[kIndex1]->GetType();
  const auto &v_type = input_args[kIndex2]->GetType();
  std::map<std::string, TypePtr> mv_args;
  (void)mv_args.insert(std::make_pair("m_type", m_type));
  (void)mv_args.insert(std::make_pair("v_type", v_type));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(mv_args, common_valid_types_with_complex, prim_name);

  const auto &var_type = input_args[kIndex0]->GetType();
  const auto &grad_type = input_args[kIndex8]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("var", var_type, {kFloat16, kFloat32}, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("gradient", grad_type, {kFloat16, kFloat32}, prim_name);

  std::map<std::string, TypePtr> args;
  (void)args.insert(std::make_pair("lr", input_args[kIndex3]->GetType()));
  (void)args.insert(std::make_pair("beta1", input_args[kIndex4]->GetType()));
  (void)args.insert(std::make_pair("beta2", input_args[kIndex5]->GetType()));
  (void)args.insert(std::make_pair("epsilon", input_args[kIndex6]->GetType()));
  (void)args.insert(std::make_pair("decay", input_args[kIndex7]->GetType()));
  (void)args.insert(std::make_pair("global_norm", input_args[kIndex9]->GetType()));
  (void)CheckAndConvertUtils::CheckTensorTypeSame(args, {kFloat32}, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type, v_type});
}
}  // namespace ops
}  // namespace mindspore
