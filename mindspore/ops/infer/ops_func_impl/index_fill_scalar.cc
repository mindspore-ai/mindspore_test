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

#include "infer/ops_func_impl/index_fill_scalar.h"
#include <memory>
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
ShapeArray IndexFillScalarFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_args) const {
  const auto &index = input_args[kIndex2];
  const auto &index_shape = index->GetShape();
  if (MS_UNLIKELY(!index->IsDynamicRank() && index_shape.size() > 1)) {
    MS_EXCEPTION(ValueError) << "'index' should be a 0 or 1-dimensional tensor, but got '" << index_shape.size() << ".";
  }

  return {input_args[kIndex0]->GetShape()};
}

std::vector<TypeId> IndexFillScalarFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_args) const {
  return {input_args[kIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
