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

#include <map>
#include <string>
#include <set>
#include "infer/ops_func_impl/bitwise_and_tensor.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
std::vector<TypeId> BitwiseAndTensorFuncImpl::InferType(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  auto x_type = input_infos[kIndex0]->GetType();
  auto other_type = input_infos[kIndex1]->GetType();
  auto end = common_float_type_ids.end();
  if (common_float_type_ids.find(x_type) != end || common_float_type_ids.find(other_type) != end) {
    MS_EXCEPTION(TypeError) << primitive->name() << " does not support floating point number.";
  }
  return {PromoteType(x_type, other_type, primitive->name())};
}

ShapeArray BitwiseAndTensorFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  const auto &x_shape = input_infos[kIndex0]->GetShape();
  const auto &other_shape = input_infos[kIndex1]->GetShape();
  return {CalBroadCastShape(x_shape, other_shape, primitive->name())};
}
}  // namespace ops
}  // namespace mindspore
