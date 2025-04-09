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

#include "infer/ops_func_impl/apply_momentum.h"

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
BaseShapePtr ApplyMomentumInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // Infer shape
  auto v_shape_ptr = input_args[kInputIndex0]->GetShape();
  auto v_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(v_shape_ptr)[kShape];
  if (IsDynamicRank(v_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }

  auto a_shape_ptr = input_args[kInputIndex1]->GetShape();
  auto a_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(a_shape_ptr)[kShape];
  if (IsDynamicRank(a_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }

  auto l_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->GetShape())[kShape];
  if (IsDynamicRank(l_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }

  auto g_shape_ptr = input_args[kInputIndex3]->GetShape();
  auto g_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(g_shape_ptr)[kShape];
  if (IsDynamicRank(g_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }

  auto m_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex4]->GetShape())[kShape];
  if (IsDynamicRank(m_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }

  if (!a_shape_ptr->IsDynamic() && !v_shape_ptr->IsDynamic()) {
    (void)CheckAndConvertUtils::CheckValue("accumulate_shape", a_shape, kEqual, "variable_shape", v_shape, prim_name);
  }
  if (!g_shape_ptr->IsDynamic() && !v_shape_ptr->IsDynamic()) {
    (void)CheckAndConvertUtils::CheckValue("gradient_shape", g_shape, kEqual, "variable_shape", v_shape, prim_name);
  }

  return v_shape_ptr->Clone();
}

TypePtr ApplyMomentumInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  // Infer type
  auto v_tensor_type = input_args[kInputIndex0]->GetType();
  auto a_tensor_type = input_args[kInputIndex1]->GetType();
  auto l_type = input_args[kInputIndex2]->GetType();
  auto g_type = input_args[kInputIndex3]->GetType();
  auto m_type = input_args[kInputIndex4]->GetType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kInt8,   kUInt8,   kInt16,     kUInt16,    kInt32,
                                         kUInt32,  kInt64,   kUInt64, kFloat64, kComplex64, kComplex128};

  (void)CheckAndConvertUtils::CheckTensorTypeValid("v_type", v_tensor_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("a_type", a_tensor_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("l_type", l_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("g_type", g_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("m_type", m_type, valid_types, prim_name);

  return v_tensor_type;
}
}  // namespace
BaseShapePtr ApplyMomentumFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyMomentumInferShape(primitive, input_args);
}

TypePtr ApplyMomentumFuncImpl::InferType(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) const {
  return ApplyMomentumInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
