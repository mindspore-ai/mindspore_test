/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/batch_norm_ext.h"

#include <memory>
#include <string>
#include <utility>

#include "abstract/dshape.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
void BatchNormExtArgShapeCheck(const PrimitivePtr &primitive, const std::string &arg_name, const InferInfoPtr &arg_info,
                               int64_t channel) {
  if (arg_info->IsNone()) {
    return;
  }
  const auto &arg_shape = arg_info->GetShape();
  MS_CHECK_VALUE(arg_shape.size() == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                          "rank of " + arg_name, SizeToLong(arg_shape.size()), kEqual, 1, primitive));
  if (MS_LIKELY(!arg_info->IsDynamic() && channel != abstract::Shape::kShapeDimAny)) {
    if (MS_UNLIKELY(arg_shape[kIndex0] != channel)) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", the " << arg_name
                               << ".shape[0] should be equal to input's channel dimension, but got "
                               << arg_shape[kIndex0] << " and " << channel;
    }
  }
}
}  // namespace
ShapeArray BatchNormExtFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &input_shape = input_infos[kIndex0]->GetShape();
  auto channel = input_infos[kIndex0]->IsDynamicRank() ? abstract::Shape::kShapeDimAny : input_shape[kIndex1];
  const std::vector<int64_t> save_mv_shape{channel};
  return {input_shape, save_mv_shape, save_mv_shape};
}

std::vector<TypeId> BatchNormExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  const auto &input_type = input_infos[kIndex0]->GetType();
  const auto &weight_info = input_infos[kIndex1];
  auto save_mv_type = weight_info->IsNone() ? input_type : weight_info->GetType();
  return {input_type, save_mv_type, save_mv_type};
}

int32_t BatchNormExtFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  constexpr auto minDim = 2;
  constexpr auto maxDim = 8;
  const auto &input_shape = input_infos[kIndex0]->GetShape();
  auto channel = abstract::Shape::kShapeDimAny;
  if (MS_LIKELY(!input_infos[kIndex0]->IsDynamicRank())) {
    MS_CHECK_VALUE(minDim <= input_shape.size() && input_shape.size() <= maxDim,
                   CheckAndConvertUtils::FormatCheckInRangeMsg("rank of input", SizeToLong(input_shape.size()),
                                                               kIncludeBoth, {minDim, maxDim}, primitive));
    channel = input_shape[kIndex1];
  }
  BatchNormExtArgShapeCheck(primitive, "weight", input_infos[kIndex1], channel);
  BatchNormExtArgShapeCheck(primitive, "bias", input_infos[kIndex2], channel);
  BatchNormExtArgShapeCheck(primitive, "running_mean", input_infos[kIndex3], channel);
  BatchNormExtArgShapeCheck(primitive, "runnning_var", input_infos[kIndex4], channel);
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
