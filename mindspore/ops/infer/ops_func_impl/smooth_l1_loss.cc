/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/smooth_l1_loss.h"
#include <algorithm>
#include <set>
#include <utility>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/ms_context.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
ShapeArray SmoothL1LossFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &prediction = input_infos[kInputIndex0];
  auto &target = input_infos[kInputIndex1];
  const auto &beta_opt = input_infos[kInputIndex2]->GetScalarValue<float>();
  if (beta_opt.has_value() && beta_opt.value() < 0) {
    MS_EXCEPTION(RuntimeError) << "For 'SmoothL1Loss'"
                               << ", the values for beta not support negative values"
                               << ", but got " << beta_opt.value() << ".";
  }
  auto prediction_shape = prediction->GetShape();
  auto target_shape = target->GetShape();
  if (!(prediction->IsDynamic() || target->IsDynamic())) {
    MS_CHECK_VALUE(
      prediction_shape == target_shape,
      CheckAndConvertUtils::FormatCheckMsg("prediction_shape", prediction_shape, kEqual, target_shape, primitive));
  }
  const auto &reduction_opt = input_infos[kInputIndex3]->GetScalarValue<int64_t>();
  if (reduction_opt.has_value()) {
    mindspore::Reduction reduction = static_cast<mindspore::Reduction>(reduction_opt.value());
    if (reduction == Reduction::NONE) {
      return {prediction_shape};
    } else {
      return {{}};
    }
  } else {
    return {{abstract::Shape::kShapeRankAny}};
  }
}

std::vector<TypeId> SmoothL1LossFuncImpl::InferType(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  auto output_type =
    PromoteType(input_infos[kInputIndex0]->GetType(), input_infos[kInputIndex1]->GetType(), "SmoothL1Loss");
  return {output_type};
}
}  // namespace ops
}  // namespace mindspore
