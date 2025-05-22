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

#include "infer/ops_func_impl/masked_scatter.h"
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
ShapeArray MaskedScatterFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto input_shape = input_infos[kInputIndex0]->GetShape();
  auto mask_shape = input_infos[kInputIndex1]->GetShape();
  auto broadcast_shape = CalBroadCastShape(input_shape, mask_shape, op_name, "input", "mask");
  return {broadcast_shape};
}

std::vector<TypeId> MaskedScatterFuncImpl::InferType(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  const auto &mask = input_infos[kInputIndex1];
  if (mask->GetType() != TypeId::kNumberTypeBool && mask->GetType() != TypeId::kNumberTypeInt8) {
    MS_EXCEPTION(TypeError) << "For " << primitive->name()
                            << ", mask should be in dtype support list [DT_UINT8,DT_BOOL,], but got " << mask->GetType()
                            << ".";
  }
  const InferInfoPtr &x = input_infos[kInputIndex0];
  auto type = x->GetType();
  return {type};
}
}  // namespace ops
}  // namespace mindspore
