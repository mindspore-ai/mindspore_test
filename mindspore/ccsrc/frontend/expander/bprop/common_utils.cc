/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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
#include "frontend/expander/bprop/common_utils.h"

#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "ops_utils/op_utils.h"

namespace mindspore::expander::bprop {
int64_t CheckRange(int64_t idx, int64_t dim_size) {
  if (idx < -dim_size || idx >= dim_size) {
    MS_EXCEPTION(IndexError) << "index {" << idx << "} is out of bounds for dimension with size {" << dim_size << "}";
  }
  return idx < 0 ? (idx + dim_size) : idx;
}

std::vector<int64_t> GetIntList(const ValuePtr &value) {
  MS_EXCEPTION_IF_NULL(value);
  if (value->isa<tensor::Tensor>()) {
    auto tensor = value->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(tensor);
    auto cpu_tensor = tensor->cpu();
    return CheckAndConvertUtils::CheckTensorIntValue("tensor", cpu_tensor, "bprop");
  } else {
    return CheckAndConvertUtils::CheckIntOrTupleInt("value", value, "bprop");
  }
}

std::vector<int64_t> GetIntList(const NodePtr &node) {
  auto value = node->BuildValue();
  MS_EXCEPTION_IF_NULL(value);
  return GetIntList(value);
}
}  // namespace mindspore::expander::bprop
