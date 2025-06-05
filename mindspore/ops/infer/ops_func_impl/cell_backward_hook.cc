/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/cell_backward_hook.h"
#include <vector>
#include <algorithm>

namespace mindspore::ops {
ShapeArray CellBackwardHookFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  ShapeArray shapes;
  shapes.reserve(input_infos.size());
  std::transform(input_infos.begin(), input_infos.end(), std::back_inserter(shapes),
                 [](const auto &input_info) { return input_info->GetShape(); });
  return shapes;
}

std::vector<TypeId> CellBackwardHookFuncImpl::InferType(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  std::vector<TypeId> types;
  types.reserve(input_infos.size());
  std::transform(input_infos.begin(), input_infos.end(), std::back_inserter(types),
                 [](const auto &input_info) { return input_info->GetType(); });
  return types;
}
}  // namespace mindspore::ops
