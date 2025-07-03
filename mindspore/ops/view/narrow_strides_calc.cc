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

#include <memory>
#include <set>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "view/narrow_strides_calc.h"

namespace {
constexpr size_t kNarrowInputsNum = 4;
}

namespace mindspore::ops {
TensorStorageInfoPtrList NarrowBasicTypeCalc(const PrimitivePtr &prim, const mindspore::tensor::TensorPtr &input_tensor,
                                             const int64_t &dim, const int64_t &start, const int64_t &length) {
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(old_tensor_info);
  auto old_shape = old_tensor_info->old_shape;

  int shape_size = SizeToLong(old_shape.size());
  MS_CHECK_VALUE(shape_size > 0, "narrow cannot be applied to a 0-dim tensor.");

  auto new_dim = DynamicDimWrap(dim, shape_size);
  auto dim_value = old_shape[new_dim];
  MS_CHECK_VALUE(start >= -dim_value && start <= dim_value,
                 "For primitive [Narrow]: start value error, start: " + std::to_string(start) +
                   ", start should be in [" + std::to_string(-dim_value) + ", " + std::to_string(dim_value) + "].");
  auto new_start = start < 0 ? start + dim_value : start;
  auto max_length = dim_value - new_start;
  MS_CHECK_VALUE(length >= 0 && length <= max_length, "length value error. length: " + std::to_string(length) +
                                                        ", length should be in [0, " + std::to_string(max_length) +
                                                        "].");
  return SliceExtBasicTypeCalc(prim, input_tensor, dim, start, new_start + length, 1);
}

TensorStorageInfoPtrList NarrowCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto dim = GetValue<int64_t>(inputs[kInputIndex1]);
  auto start = GetValue<int64_t>(inputs[kInputIndex2]);
  auto length = GetValue<int64_t>(inputs[kInputIndex3]);
  return NarrowBasicTypeCalc(prim, input_tensor, dim, start, length);
}

REG_VIEW_STRIDES_CALC_FUN(Narrow, NarrowCalc);
}  // namespace mindspore::ops
