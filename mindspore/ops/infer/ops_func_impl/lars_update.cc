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

#include "infer/ops_func_impl/lars_update.h"

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
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
namespace {
BaseShapePtr LARSUpdateInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();

  auto weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex0]->GetShapeTrack());
  auto gradient_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex1]->GetShapeTrack());
  auto norm_weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex2]->GetShapeTrack());
  auto norm_gradient_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex3]->GetShapeTrack());
  auto weight_decay_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex4]->GetShapeTrack());
  auto learning_rate_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex5]->GetShapeTrack());

  if (weight_shape[kShape].size() != gradient_shape[kShape].size()) {
    MS_EXCEPTION(ValueError) << "For '" << op_name
                             << "', weight shape size must be equal to gradient shape size, but got "
                             << "weight shape size: " << weight_shape[kShape].size()
                             << ", gradient shape size: " << gradient_shape[kShape].size() << ".";
  }
  if (norm_weight_shape[kShape].size() != norm_gradient_shape[kShape].size()) {
    MS_EXCEPTION(ValueError) << "For " << op_name
                             << "', norm weight shape size must be equal to norm gradient shape size, but got "
                             << "norm weight shape size: " << norm_weight_shape[kShape].size()
                             << ", norm gradient shape size: " << norm_gradient_shape[kShape].size() << ".";
  }
  for (size_t index = 0; index < weight_shape[kShape].size(); index++) {
    if (weight_shape[kShape][index] != gradient_shape[kShape][index]) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the " << index
                               << "th dim of weight shape and gradient shape must be equal, but got "
                               << "weight shape[" << index << "]: " << weight_shape[kShape][index]
                               << ", gradient shape[" << index << "]: " << gradient_shape[kShape][index] << ".";
    }
  }
  for (size_t index = 0; index < norm_weight_shape[kShape].size(); index++) {
    if (norm_weight_shape[kShape][index] != norm_gradient_shape[kShape][index]) {
      MS_EXCEPTION(ValueError) << "For '" << op_name << "', the " << index
                               << "th dim of norm weight shape and norm gradient shape must be equal, but got "
                               << "norm weight shape[" << index << "]: " << norm_weight_shape[kShape][index]
                               << ", norm gradient shape[" << index << "]: " << norm_gradient_shape[kShape][index]
                               << ".";
    }
  }
  auto shp_len = weight_decay_shape[kShape].size();
  auto para_name = input_args[kIndex4]->ToString();
  (void)CheckAndConvertUtils::CheckInteger(para_name, SizeToLong(shp_len), kLessEqual, 1);
  if (shp_len == 1) {
    (void)CheckAndConvertUtils::CheckInteger(para_name, weight_decay_shape[kShape][0], kEqual, 1);
  }
  shp_len = learning_rate_shape[kShape].size();
  para_name = input_args[kIndex5]->ToString();
  (void)CheckAndConvertUtils::CheckInteger(para_name, SizeToLong(shp_len), kLessEqual, 1);
  if (shp_len == 1) {
    (void)CheckAndConvertUtils::CheckInteger(para_name, learning_rate_shape[kShape][0], kEqual, 1);
  }

  return std::make_shared<abstract::Shape>(weight_shape[kShape]);
}

TypePtr LARSUpdateInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("Weight dtype", input_args[kIndex0]->GetType());
  (void)types.emplace("gradient dtype", input_args[kIndex1]->GetType());
  (void)types.emplace("norm weight dtype", input_args[kIndex2]->GetType());
  (void)types.emplace("norm gradient dtype", input_args[kIndex3]->GetType());
  const std::set<TypePtr> valid_types = {kInt16, kInt32, kFloat16, kFloat32};
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, primitive->name());
  return types["Weight dtype"];
}
}  // namespace
BaseShapePtr LARSUpdateFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  return LARSUpdateInferShape(primitive, input_args);
}

TypePtr LARSUpdateFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  return LARSUpdateInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore