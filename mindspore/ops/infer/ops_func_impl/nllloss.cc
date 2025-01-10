/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/nllloss.h"
#include <set>
#include <algorithm>
#include <memory>
#include <string>
#include "abstract/dshape.h"
#include "mindapi/base/types.h"
#include "mindspore/ops/op_def/op_name.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops_utils/op_constants.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
void CheckNLLLossShapeValid(const PrimitivePtr &primitive, const ShapeVector &logits_shape,
                            const ShapeVector &labels_shape, const ShapeVector &weight_shape) {
  if (logits_shape.size() == 1) {
    if (logits_shape[0] > abstract::Shape::kShapeDimAny) {
      if (labels_shape[0] > abstract::Shape::kShapeDimAny) {
        MS_CHECK_VALUE(labels_shape[0] == 1, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                               "labels shape", labels_shape[0], kEqual, 1, primitive));
      }
      if (weight_shape[0] > abstract::Shape::kShapeDimAny) {
        MS_CHECK_VALUE(weight_shape[0] == logits_shape[0],
                       CheckAndConvertUtils::FormatCheckIntegerMsg("weight shape", weight_shape[0], kEqual,
                                                                   logits_shape[0], primitive));
      }
    }
  } else if (logits_shape.size() == 2) {
    if (logits_shape[0] > abstract::Shape::kShapeDimAny && labels_shape[0] > abstract::Shape::kShapeDimAny) {
      MS_CHECK_VALUE(labels_shape[0] == logits_shape[0],
                     CheckAndConvertUtils::FormatCheckIntegerMsg("labels shape", labels_shape[0], kEqual,
                                                                 logits_shape[0], primitive));
    }
    if (logits_shape[1] > abstract::Shape::kShapeDimAny && weight_shape[0] > abstract::Shape::kShapeDimAny) {
      MS_CHECK_VALUE(weight_shape[0] == logits_shape[1],
                     CheckAndConvertUtils::FormatCheckIntegerMsg("weight shape", weight_shape[0], kEqual,
                                                                 logits_shape[1], primitive));
    }
  }
}

ShapeArray NLLLossFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto target_shape = input_infos[kInputIndex1]->GetShape();

  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  std::string device_target = context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto jit_level = context_ptr->GetJitLevel();
  auto execution_mode = context_ptr->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (device_target == kGPUDevice || device_target == kCPUDevice ||
      (execution_mode == kGraphMode && jit_level == "O2")) {
    auto logits_shape = input_infos[kInputIndex0]->GetShape();
    auto weight_shape = input_infos[kInputIndex2]->GetShape();

    const size_t x_rank = 1;
    const size_t DIM_2 = 2;
    MS_CHECK_VALUE(target_shape.size() == x_rank, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                                    "target_rank", target_shape.size(), kEqual, x_rank, primitive));
    MS_CHECK_VALUE(weight_shape.size() == x_rank, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                                    "weight_rank", weight_shape.size(), kEqual, x_rank, primitive));
    MS_CHECK_VALUE(logits_shape.size() >= x_rank && logits_shape.size() <= DIM_2,
                   CheckAndConvertUtils::FormatCheckInRangeMsg("logits_shape_rank", logits_shape.size(), kIncludeBoth,
                                                               {1, 2}, primitive));
    CheckNLLLossShapeValid(primitive, logits_shape, target_shape, weight_shape);
  }

  ShapeVector weight_out_shape = {};
  auto reduction = static_cast<Reduction>(input_infos[kInputIndex3]->GetScalarValue<int64_t>().value());
  if (reduction == Reduction::NONE) {
    if (target_shape.size() == kDim1 && target_shape[0] == abstract::TensorShape::kShapeRankAny) {
      target_shape = {-1};
    }
    return {target_shape, weight_out_shape};
  } else {
    ShapeVector out_shape = {};
    return {out_shape, weight_out_shape};
  }
}

std::vector<TypeId> NLLLossFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto input_data_type = input_infos[kInputIndex0]->GetType();
  return {input_data_type, input_data_type};
}
}  // namespace ops
}  // namespace mindspore
