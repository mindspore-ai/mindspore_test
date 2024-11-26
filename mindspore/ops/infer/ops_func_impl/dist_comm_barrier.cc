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

#include "infer/ops_func_impl/dist_comm_barrier.h"
#include <memory>
#include "ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
ShapeArray DistCommBarrierFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  return {ShapeVector{}};
}

std::vector<TypeId> DistCommBarrierFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  return {kTypeNone->type_id()};
}
}  // namespace ops
}  // namespace mindspore
