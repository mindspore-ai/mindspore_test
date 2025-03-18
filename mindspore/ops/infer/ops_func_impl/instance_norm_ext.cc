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

#include "infer/ops_func_impl/instance_norm_ext.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {

ShapeArray InstanceNormExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto &input_x = input_infos[kIndex0];
  auto input_x_shape = input_x->GetShape();
  if (!IsDynamic(input_x_shape)) {
    auto input_x_rank = SizeToLong(input_x_shape.size());
    if (input_x_rank < 2 || input_x_rank > 8) {
      MS_EXCEPTION(ValueError) << "For primitive[InstanceNorm], the dims of input should "
                               << "between 2 and 8 dimensional, but got " << input_x_rank << "-dimensional.";
    }
  }
  return {input_x_shape};
}

std::vector<TypeId> InstanceNormExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  TypeId input_type = input_infos[kIndex0]->GetType();
  return {input_type};
}
}  // namespace ops
}  // namespace mindspore
