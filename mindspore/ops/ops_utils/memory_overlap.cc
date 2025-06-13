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
#include "ops_utils/memory_overlap.h"
#include "ir/tensor.h"

namespace mindspore {
MemOverlap IsInternalOverlap(const TensorPtr &variable_tensor) {
  // For tensor is not type of view, never has overlap in tensor.
  if (variable_tensor->storage_info() == nullptr) {
    return MemOverlap::No;
  }

  // For broadcast_to case, there is overlap in tensor of course.
  const auto &strides = variable_tensor->storage_info()->strides;
  const auto &shape = variable_tensor->storage_info()->shape;
  if (strides.size() != shape.size()) {
    MS_LOG(EXCEPTION) << "Size of strides and shape are not equal:" << strides.size() << ", " << shape.size();
  }
  for (uint32_t i = 0; i < strides.size(); i++) {
    if (shape[i] > 1 && strides[i] == 0) {
      return MemOverlap::Yes;
    }
  }

  // Others, to hard to judge.
  return MemOverlap::TooHard;
}

// This function used to assert the input of an inplace operator,
// for which case the result is uncertain. In the follow case, value of d is uncertain.
// a = [[1], [2], [3]]
// b = mint.broadcast_to(a, (3,2))
// c = [[1, 2], [1, 2], [1, 2]]
// d = b.copy_ext(c)
void ThrowExpectionWhenInternalOverlap(const TensorPtr &variable_tensor) {
  if (IsInternalOverlap(variable_tensor) == MemOverlap::Yes) {
    MS_LOG(EXCEPTION) << "This tensor has multi element reference to the same memory address,"
                         "which is forbidden.You can clone it before execute the operation.";
  }
}
}  // namespace mindspore
