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

#include "infer/ops_func_impl/empty_like.h"
#include <vector>
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
ShapeArray EmptyLikeFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetShape()};
}

std::vector<TypeId> EmptyLikeFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  const auto &dtype_info = input_infos[kInputIndex1];
  // return dtype if dtype_info is not none
  if (!dtype_info->IsNone()) {
    const auto dtype_opt = dtype_info->GetScalarValue<int64_t>();
    if (dtype_opt.has_value()) {
      return {static_cast<TypeId>(dtype_opt.value())};
    }
  }
  // return input tensor's dtype if dtype_info is none
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
