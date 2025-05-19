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
#include "infer/ops_func_impl/remainder_tensor_scalar.h"
#include "ops_utils/op_utils.h"

namespace mindspore::ops {
ShapeArray RemainderTensorScalarFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<InferInfoPtr> &input_infos) const {
  return {input_infos[kIndex0]->GetShape()};
}

std::vector<TypeId> RemainderTensorScalarFuncImpl::InferType(const PrimitivePtr &primitive,
                                                             const std::vector<InferInfoPtr> &input_infos) const {
  const auto &x_type = input_infos[kIndex0]->GetType();
  const auto &other_type = input_infos[kIndex1]->GetType();
  if (common_float_type_ids.count(x_type) == 0 && common_float_type_ids.count(other_type) != 0) {
    return {kNumberTypeFloat32};
  }
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace mindspore::ops
