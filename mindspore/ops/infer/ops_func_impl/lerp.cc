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

#include "infer/ops_func_impl/lerp.h"
#include <set>
#include <memory>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore {
namespace ops {

ShapeArray LerpFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto start_shape = input_infos[kInputIndex0]->GetShape();
  auto end_shape = input_infos[kInputIndex1]->GetShape();
  auto weight_shape = input_infos[kInputIndex2]->GetShape();

  auto broadcast_shape = CalBroadCastShape(start_shape, end_shape, op_name, "start", "end");

  (void)CalBroadCastShape(start_shape, weight_shape, op_name, "start", "weight");
  (void)CalBroadCastShape(end_shape, weight_shape, op_name, "end", "weight");

  broadcast_shape = CalBroadCastShape(broadcast_shape, weight_shape, op_name);
  return ShapeArray{broadcast_shape};
}

std::vector<TypeId> LerpFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
