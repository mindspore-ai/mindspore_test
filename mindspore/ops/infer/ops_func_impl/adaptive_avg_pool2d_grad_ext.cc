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
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "infer/ops_func_impl/adaptive_avg_pool2d_grad_ext.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
ShapeArray AdaptiveAvgPool2DGradExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  auto &x_tensor = input_infos[kInputIndex1];
  auto x_shape = x_tensor->GetShape();
  if (!x_tensor->IsDynamicRank()) {
    const int64_t orig_input_shape_shape = SizeToLong(x_shape.size());
    CheckAndConvertUtils::CheckInRange("length of orig_input_shape", orig_input_shape_shape, kIncludeBoth, {3, 4},
                                       kNameAdaptiveAvgPool2DGradExt);
  }
  return {x_shape};
}

std::vector<TypeId> AdaptiveAvgPool2DGradExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                                const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
