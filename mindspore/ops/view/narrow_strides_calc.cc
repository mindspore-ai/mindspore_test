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

namespace mindspore::ops {
TensorStorageInfoPtrList NarrowBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor, const int64_t &dim,
                                             const int64_t &start, const int64_t &length) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &input_shape = input_tensor->shape();

  int input_dim = SizeToLong(input_shape.size());
  MS_CHECK_VALUE(input_dim > 0, "narrow cannot be applied to a 0-dim tensor.");

  auto dim_value = input_shape[DynamicDimWrap(dim, input_dim)];
  MS_CHECK_VALUE(start >= -dim_value && start <= dim_value,
                 "For primitive [Narrow]: start value error, start: " + std::to_string(start) +
                   ", start should be in [" + std::to_string(-dim_value) + ", " + std::to_string(dim_value) + "].");
  auto new_start = start < 0 ? start + dim_value : start;

  auto max_length = dim_value - new_start;
  MS_CHECK_VALUE(length >= 0 && length <= max_length,
                 "For 'Narrow', start (" + std::to_string(start) + "), + length (" + std::to_string(length) +
                   ") exceeds dimension size (" + std::to_string(dim_value) + ").");
  return SliceExtBasicTypeCalc(input_tensor, dim, new_start, new_start + length, 1);
}
}  // namespace mindspore::ops
