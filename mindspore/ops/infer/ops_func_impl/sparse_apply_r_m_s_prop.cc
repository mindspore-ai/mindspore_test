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

#include "infer/ops_func_impl/sparse_apply_r_m_s_prop.h"

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
BaseShapePtr SparseApplyRMSPropInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto var_shape_ptr = input_args[0]->GetShape();
  auto ms_shape_ptr = input_args[1]->GetShape();
  auto mom_shape_ptr = input_args[2]->GetShape();

  auto var_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(var_shape_ptr)[kShape];
  auto ms_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(ms_shape_ptr)[kShape];
  auto mom_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(mom_shape_ptr)[kShape];
  auto lr_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[3]->GetShape())[kShape];
  auto lr_shape_rank = SizeToLong(lr_shape.size());
  auto grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[4]->GetShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[5]->GetShape())[kShape];

  // Args lr must be scalar
  const int64_t input_num = 0;
  if (!IsDynamic(lr_shape)) {
    (void)CheckAndConvertUtils::CheckInteger("size of lr_shape", lr_shape_rank, kEqual, input_num, primitive->name());
  }

  std::vector<ShapeVector> check_shapes = {ms_shape, mom_shape, grad_shape, var_shape};
  auto is_dynamic = std::any_of(check_shapes.begin(), check_shapes.end(), IsDynamic);
  if (!is_dynamic) {
    // Shape of var、ms、mom、grad must be same
    std::map<std::string, ShapeVector> same_shape_args_map;
    (void)same_shape_args_map.insert(std::make_pair("shape of ms ", ms_shape));
    (void)same_shape_args_map.insert(std::make_pair("shape of mom ", mom_shape));
    (void)same_shape_args_map.insert(std::make_pair("shape of grad ", grad_shape));
    for (auto &elem : same_shape_args_map) {
      CheckAndConvertUtils::Check(elem.first, elem.second, kEqual, var_shape, prim_name);
    }
  }

  // Indices must be rank 1
  const int64_t input_num1 = 1;
  (void)CheckAndConvertUtils::CheckInteger("indices dim", SizeToLong(indices_shape.size()), kEqual, input_num1,
                                           prim_name);

  // Dimension of var must be equal or greater than 1
  (void)CheckAndConvertUtils::CheckInteger("dimension of var", SizeToLong(var_shape.size()), kGreaterEqual, input_num1,
                                           prim_name);

  // Indices shape must be equal to the first dimension of var
  if (!(IsDynamic(indices_shape) || IsDynamic(var_shape))) {
    CheckAndConvertUtils::Check("indices shape", indices_shape[0], kEqual, var_shape[0], prim_name);
  }

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{var_shape_ptr, ms_shape_ptr, mom_shape_ptr});
}

TuplePtr SparseApplyRMSPropInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();

  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  if (!input_args[kInputIndex4]->GetType()->isa<TensorType>() ||
      !input_args[kInputIndex5]->GetType()->isa<TensorType>()) {
    MS_EXCEPTION(TypeError) << "For SparseApplyRMSProp, 'grad' or 'indices' should be Tensor.";
  }
  auto var_type = input_args[kInputIndex0]->GetType();
  auto ms_type = input_args[kInputIndex1]->GetType();
  auto mom_type = input_args[kInputIndex2]->GetType();
  auto lr_type = input_args[kInputIndex3]->GetType();
  auto grad_type = input_args[kInputIndex4]->GetType();
  auto indices_type = input_args[kInputIndex5]->GetType();

  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("var", var_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("ms", ms_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("mom", mom_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("grad", grad_type, valid_types, prim_name);
  (void)CheckAndConvertUtils::CheckTensorTypeValid("lr", lr_type, valid_types, prim_name);

  const std::set<TypePtr> valid_types1 = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("indices", indices_type, valid_types1, prim_name);

  return std::make_shared<Tuple>(std::vector<TypePtr>{var_type, ms_type, mom_type});
}
}  // namespace

BaseShapePtr SparseApplyRMSPropFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) const {
  return SparseApplyRMSPropInferShape(primitive, input_args);
}

TypePtr SparseApplyRMSPropFuncImpl::InferType(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  return SparseApplyRMSPropInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
