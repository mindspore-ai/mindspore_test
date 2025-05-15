/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#include "infer/ops_func_impl/batch_norm_grad_ext.h"

#include <memory>

#include "abstract/dshape.h"

namespace mindspore {
namespace ops {
ShapeArray BatchNormGradExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  const auto &input_shape = input_infos[kIndex1]->GetShape();
  auto channel = input_infos[kIndex1]->IsDynamicRank() ? abstract::Shape::kShapeDimAny : input_shape[kIndex1];
  const std::vector<int64_t> weight_bias_shape{channel};
  return {input_shape, weight_bias_shape, weight_bias_shape};
}

std::vector<TypeId> BatchNormGradExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType(), kNumberTypeFloat32, kNumberTypeFloat32};
}
}  // namespace ops
}  // namespace mindspore
