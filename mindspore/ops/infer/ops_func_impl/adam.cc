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

#include "infer/ops_func_impl/adam.h"

#include <memory>
#include <map>
#include <set>
#include <string>
#include <vector>
#include <functional>

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
BaseShapePtr AdamFuncImpl::InferShape(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto var_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto m_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto v_shape_ptr = input_args[kInputIndex2]->GetShape();
  auto grad_shape_ptr = input_args[kInputIndex9]->GetShape();
  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->GetShape())[kShape];
  auto m_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->GetShape())[kShape];
  auto v_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShape())[kShape];
  auto beta1_power_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->GetShape())[kShape];
  auto beta2_power_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->GetShape())[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex5]->GetShape())[kShape];
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex9]->GetShape())[kShape];
  if (IsDynamicRank(var_shape) || IsDynamicRank(m_shape) || IsDynamicRank(v_shape)) {
    auto unknow_shape_ptr = std::make_shared<abstract::Shape>(std::vector<int64_t>{abstract::Shape::kShapeRankAny});
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{unknow_shape_ptr, unknow_shape_ptr, unknow_shape_ptr});
  }
  MS_EXCEPTION_IF_NULL(var_shape_ptr);
  MS_EXCEPTION_IF_NULL(m_shape_ptr);
  MS_EXCEPTION_IF_NULL(v_shape_ptr);
  MS_EXCEPTION_IF_NULL(grad_shape_ptr);
  if (var_shape_ptr->IsDynamic() || m_shape_ptr->IsDynamic() || v_shape_ptr->IsDynamic() ||
      grad_shape_ptr->IsDynamic()) {
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{var_shape_ptr, m_shape_ptr, v_shape_ptr});
  }
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, m_shape, prim_name);
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, v_shape, prim_name);
  CheckAndConvertUtils::Check("var_shape", var_shape, kEqual, grad_shape, prim_name);

  int64_t batch_rank = 0;
  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }
  if (batch_rank != 0) {
    (void)CheckAndConvertUtils::CheckInteger("beta1_power_shape size", SizeToLong(beta1_power_shape.size()), kEqual,
                                             batch_rank, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("beta2_power_shape size", SizeToLong(beta2_power_shape.size()), kEqual,
                                             batch_rank, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("lr_shape size", SizeToLong(lr_shape.size()), kEqual, batch_rank,
                                             prim_name);
  } else {
    if (beta1_power_shape.size() == 1 || beta1_power_shape.size() == 0) {
      (void)CheckAndConvertUtils::CheckInteger(
        "beta1_power_shape element num",
        std::accumulate(beta1_power_shape.begin(), beta1_power_shape.end(), 1, std::multiplies<int>()), kEqual, 1,
        prim_name);
    } else {
      MS_EXCEPTION(ValueError) << "The rank of beta1_power must be 0 or 1 but got " << lr_shape.size();
    }
    if (beta2_power_shape.size() == 1 || beta2_power_shape.size() == 0) {
      (void)CheckAndConvertUtils::CheckInteger(
        "beta2_power_shape element num",
        std::accumulate(beta2_power_shape.begin(), beta2_power_shape.end(), 1, std::multiplies<int>()), kEqual, 1,
        prim_name);
    } else {
      MS_EXCEPTION(ValueError) << "The rank of beta2_power must be 0 or 1 but got " << lr_shape.size();
    }
    if (lr_shape.size() == 1 || lr_shape.size() == 0) {
      (void)CheckAndConvertUtils::CheckInteger(
        "lr_shape element num", std::accumulate(lr_shape.begin(), lr_shape.end(), 1, std::multiplies<int>()), kEqual, 1,
        prim_name);
    } else {
      MS_EXCEPTION(ValueError) << "The rank of lr must be 0 or 1 but got " << lr_shape.size();
    }
  }

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape_ptr, m_shape_ptr, v_shape_ptr});
}

TypePtr AdamFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  auto prim_name = primitive->name();
  auto var_type = input_args[kInputIndex0]->GetType();
  auto m_type = input_args[kInputIndex1]->GetType();
  auto v_type = input_args[kInputIndex2]->GetType();
  auto grad_type = input_args[kInputIndex9]->GetType();

  std::set<TypePtr> num_type = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("var", var_type, num_type, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grad", grad_type, num_type, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, m_type, v_type});
}
}  // namespace ops
}  // namespace mindspore
