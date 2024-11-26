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

#include "infer/ops_func_impl/dist_comm_gather_into_tensor.h"
#include <memory>
#include <set>
#include <string>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "infer/ops_func_impl/op_comm_func_impl.h"

namespace mindspore {
namespace ops {
ShapeArray DistCommGatherIntoTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  auto &value = input_infos[kIndex2];
  auto rank_size = CheckRankSize(primitive->name(), value);

  auto input_shape = input_infos[kIndex1]->GetShape();
  input_shape[kIndex0] = input_shape[kIndex0] * rank_size;
  auto dst_rank = GetRankValue(primitive->name(), input_infos[kIndex3]);
  auto local_rank = GetRankValue(primitive->name(), input_infos[kIndex4]);
  if (dst_rank == local_rank) {
    const auto &output_shape = input_infos[kIndex0]->GetShape();
    CheckInferShape(primitive->name(), input_shape, output_shape);
  }

  return {input_shape};
}

std::vector<TypeId> DistCommGatherIntoTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                                                const InferInfoPtrList &input_infos) const {
  auto type = input_infos[kIndex1]->GetType();
  auto out_type = input_infos[kIndex0]->GetType();
  return {CheckInferTypes(primitive->name(), type, out_type)};
}
}  // namespace ops
}  // namespace mindspore
