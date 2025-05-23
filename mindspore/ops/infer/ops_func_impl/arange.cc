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

#include "infer/ops_func_impl/arange.h"

#include <any>
#include <memory>
#include <vector>
#include <utility>
#include <algorithm>

#include "abstract/dshape.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore::ops {
namespace {
bool CheckDtypeValidAndIsInteger(const PrimitivePtr &primitive, const InferInfoPtr &dtype_info) {
  if (dtype_info->IsNone()) {
    return false;
  }
  auto dtype_opt = dtype_info->GetScalarValue<int64_t>();
  MS_CHECK_VALUE(dtype_opt.has_value(), primitive->name() + " error: the dtype argument has no valid value.");
  auto dtype_id = static_cast<TypeId>(dtype_opt.value());
  if (dtype_id == kNumberTypeFloat16 || dtype_id == kNumberTypeFloat32 || dtype_id == kNumberTypeFloat64 ||
      dtype_id == kNumberTypeBFloat16) {
    return false;
  }
  if (dtype_id == kNumberTypeInt32 || dtype_id == kNumberTypeInt64) {
    return true;
  }
  MS_EXCEPTION(ValueError) << "For Arange, the dtype argument must be: "
                           << "int32, int64, float16, float32, float64, or bfloat16, but got "
                           << TypeIdToString(dtype_id) << ".";
}

template <typename T>
int64_t ComputeShapeSize(const InferInfoPtrList &input_infos, bool result_type_is_int) {
  auto start_opt = input_infos[kIndex0]->GetScalarValue<T>();
  auto end_opt = input_infos[kIndex1]->GetScalarValue<T>();
  auto step_opt = input_infos[kIndex2]->GetScalarValue<T>();

  if (MS_UNLIKELY(!start_opt.has_value()) || MS_UNLIKELY(!end_opt.has_value()) || MS_UNLIKELY(!step_opt.has_value())) {
    return abstract::Shape::kShapeDimAny;
  }

  auto start = start_opt.value();
  auto end = end_opt.value();
  auto step = step_opt.value();

  bool step_not_zero = static_cast<bool>(step);
  bool step_positive;
  if constexpr (std::is_same<T, bool>::value) {
    step_positive = step_not_zero;
  } else {
    step_positive = step > 0;
  }
  bool step_negative = !step_positive && step_not_zero;

  if (!step_not_zero) {
    MS_EXCEPTION(ValueError) << "For Arange, step must not be zero.";
  }
  if (step_positive && start > end) {
    MS_EXCEPTION(ValueError) << "For Arange, step cannot be positive when end < start.";
  }
  if (step_negative && start < end) {
    MS_EXCEPTION(ValueError) << "For Arange, step cannot be negative when end > start.";
  }

  double shape_size = 0;
  if (!result_type_is_int) {
    shape_size = std::ceil((end - start) / static_cast<double>(step));
  } else {
    shape_size = std::ceil(static_cast<double>(static_cast<int64_t>(end) - static_cast<int64_t>(start)) /
                           static_cast<int64_t>(step));
  }

  return static_cast<int64_t>(shape_size);
}

int64_t GetShapeSize(const InferInfoPtrList &input_infos, bool result_type_is_int) {
  int64_t shape_size{abstract::Shape::kShapeDimAny};
  auto start_type = input_infos[kIndex0]->GetType();
  switch (start_type) {
    case kNumberTypeBool:
      shape_size = ComputeShapeSize<bool>(input_infos, result_type_is_int);
      break;
    case kNumberTypeInt32:
    case kNumberTypeInt:
      shape_size = ComputeShapeSize<int32_t>(input_infos, result_type_is_int);
      break;
    case kNumberTypeInt64:
      shape_size = ComputeShapeSize<int64_t>(input_infos, result_type_is_int);
      break;
    case kNumberTypeFloat:
    case kNumberTypeFloat32:
      shape_size = ComputeShapeSize<float>(input_infos, result_type_is_int);
      break;
    case kNumberTypeFloat64:
      shape_size = ComputeShapeSize<double>(input_infos, result_type_is_int);
      break;
    default:
      MS_EXCEPTION(TypeError)
        << "For Arange, the type of input must be int32, int64, float32, float64 or bool, but got "
        << TypeIdToString(start_type) << ".";
  }
  return shape_size;
}
}  // namespace

ShapeArray ArangeFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &dtype_info = input_infos[kIndex3];
  auto result_type_is_int = CheckDtypeValidAndIsInteger(primitive, dtype_info);
  auto shape_size = GetShapeSize(input_infos, result_type_is_int);
  ShapeVector output_shape{shape_size};
  return {std::move(output_shape)};
}

TypeIdList ArangeFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &dtype_info = input_infos[kIndex3];
  if (dtype_info->IsNone()) {
    auto CheckFloatFunc = [](const InferInfoPtr &input_info) {
      auto type_id = input_info->GetType();
      return type_id == kNumberTypeFloat64 || type_id == kNumberTypeFloat32;
    };
    if (std::any_of(input_infos.begin(), input_infos.begin() + kIndex3, CheckFloatFunc)) {
      return {kNumberTypeFloat32};
    }
    return {kNumberTypeInt64};
  }
  auto dtype_opt = dtype_info->GetScalarValue<int64_t>();
  MS_CHECK_VALUE(dtype_opt.has_value(), primitive->name() + " error: the dtype argument has no valid value.");
  return {static_cast<TypeId>(dtype_opt.value())};
}
}  // namespace mindspore::ops
