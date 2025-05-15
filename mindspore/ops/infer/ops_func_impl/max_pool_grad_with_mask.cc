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

#include "infer/ops_func_impl/max_pool_grad_with_mask.h"

#include <utility>
#include <algorithm>
#include <memory>

#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/ms_context.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
ShapeArray MaxPoolGradWithMaskFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  const auto &x_info = input_infos[kIndex0];
  if (x_info->IsDynamicRank()) {
    auto output_shape = std::vector<int64_t>(kIndex4, abstract::Shape::kShapeDimAny);
    return {std::move(output_shape)};
  }
  return {x_info->GetShape()};
}

TypeIdList MaxPoolGradWithMaskFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
