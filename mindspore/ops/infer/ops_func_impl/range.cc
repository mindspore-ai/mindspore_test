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

#include "infer/ops_func_impl/range.h"

#include <memory>
#include <vector>
#include <utility>
#include <algorithm>

#include "abstract/dshape.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore::ops {
template <typename T>
ShapeVector CalculateShapeSize(const InferInfoPtr &start_info, const InferInfoPtr &limit_info,
                               const InferInfoPtr &delta_info) {
  ShapeVector out_shape = {};
  auto start_opt = start_info->GetScalarValue<T>();
  auto limit_opt = limit_info->GetScalarValue<T>();
  auto delta_opt = delta_info->GetScalarValue<T>();

  if (MS_UNLIKELY(!start_opt.has_value()) || MS_UNLIKELY(!limit_opt.has_value()) ||
      MS_UNLIKELY(!delta_opt.has_value())) {
    (void)out_shape.emplace_back(abstract::Shape::kShapeDimAny);
    return out_shape;
  }

  auto start = start_opt.value();
  auto limit = limit_opt.value();
  auto delta = delta_opt.value();

  if (delta == T(0)) {
    MS_EXCEPTION(ValueError) << "For Range, delta cannot be equal to zero.";
  }
  if (delta > 0 && start > limit) {
    MS_EXCEPTION(ValueError) << "For Range, delta cannot be positive when limit < start.";
  }
  if (delta < 0 && start < limit) {
    MS_EXCEPTION(ValueError) << "For Range, delta cannot be negative when limit > start.";
  }

  int64_t shape_size = 0;
  if (std::is_integral<T>::value) {
    shape_size = static_cast<int64_t>((std::abs(limit - start) + std::abs(delta) - 1) / std::abs(delta));
  } else {
    shape_size = static_cast<int64_t>(std::ceil(std::abs((limit - start) / delta)));
  }

  if (shape_size < 0) {
    MS_EXCEPTION(ValueError) << "For Range, infer shape error, shape_size [" << shape_size << "] is negative.";
  }

  (void)out_shape.emplace_back(shape_size);
  return out_shape;
}

ShapeArray RangeFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &start_info = input_infos[kIndex0];
  const auto &limit_info = input_infos[kIndex1];
  const auto &delta_info = input_infos[kIndex2];
  ShapeVector output_shape;
  auto start_type = start_info->GetType();
  switch (start_type) {
    case kNumberTypeInt32:
      output_shape = CalculateShapeSize<int32_t>(start_info, limit_info, delta_info);
      break;
    case kNumberTypeInt64:
      output_shape = CalculateShapeSize<int64_t>(start_info, limit_info, delta_info);
      break;
    case kNumberTypeFloat32:
      output_shape = CalculateShapeSize<float>(start_info, limit_info, delta_info);
      break;
    case kNumberTypeFloat64:
      output_shape = CalculateShapeSize<double>(start_info, limit_info, delta_info);
      break;
    default:
      MS_EXCEPTION(TypeError) << "For Range, the dtype of input must be int64 or float64, but got "
                              << TypeIdToString(start_type) << ".";
  }

  return {std::move(output_shape)};
}

TypeIdList RangeFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto start_type = input_infos[kIndex0]->GetType();
  auto limit_type = input_infos[kIndex1]->GetType();
  auto delta_type = input_infos[kIndex2]->GetType();
  if (!(start_type == limit_type && limit_type == delta_type)) {
    MS_EXCEPTION(TypeError) << "For Range, the dtypes of inputs must be the same, but got ("
                            << TypeIdToString(start_type) << ", " << TypeIdToString(limit_type) << ", "
                            << TypeIdToString(delta_type) << ").";
  }
  if (start_type == kNumberTypeFloat32 || start_type == kNumberTypeFloat64) {
    return {kNumberTypeFloat32};
  }
  return {start_type};
}

int32_t RangeFuncImpl::CheckValidation(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto maxlen_opt = input_infos[kIndex3]->GetScalarValue<int64_t>();
  if (MS_UNLIKELY(!maxlen_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  MS_CHECK_VALUE(maxlen_opt.value() > 0,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("maxlen", maxlen_opt.value(), kGreaterThan, 0, primitive));
  return OP_CHECK_SUCCESS;
}
}  // namespace mindspore::ops
