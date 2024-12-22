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

TensorStorageInfoPtrList NarrowCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);

  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(old_tensor_info);
  auto old_shape = old_tensor_info->old_shape;

  int shape_size = SizeToLong(old_shape.size());
  MS_CHECK_VALUE(shape_size > 0, "narrow cannot be applied to a 0-dim tensor.");

  auto dim = GetValue<int64_t>(inputs[kInputIndex1]);
  dim = DynamicDimWrap(dim, shape_size);
  auto dim_value = old_shape[dim];
  auto start = GetValue<int64_t>(inputs[kInputIndex2]);
  MS_CHECK_VALUE(start >= -dim_value && start <= dim_value,
                 "For primitive [Narrow]: start value error, start: " + std::to_string(start) +
                   ", start should be in [" + std::to_string(-dim_value) + ", " + std::to_string(dim_value) + "].");
  start = start < 0 ? start + dim_value : start;

  auto length = GetValue<int64_t>(inputs[kInputIndex3]);
  auto max_length = dim_value - start;
  MS_CHECK_VALUE(length >= 0 && length <= max_length, "length value error. length: " + std::to_string(length) +
                                                        ", length should be in [0, " + std::to_string(max_length) +
                                                        "].");

  auto new_inputs = inputs;
  new_inputs.pop_back();
  new_inputs.emplace_back(MakeValue<int64_t>(start + length));
  new_inputs.emplace_back(MakeValue<int64_t>(1));
  return SliceExtCalc(prim, new_inputs);
}

REG_VIEW_STRIDES_CALC_FUN(Narrow, NarrowCalc);
}  // namespace mindspore::ops
