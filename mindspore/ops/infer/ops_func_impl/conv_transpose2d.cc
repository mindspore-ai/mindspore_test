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

#include "infer/ops_func_impl/conv_transpose2d.h"

#include <utility>
#include <string>
#include <set>

#include "utils/check_convert_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray ConvTranspose2DFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  const auto &input_info = input_infos[idxes_.input_idx];
  const auto &input_shape = input_info->GetShape();
  if (input_info->IsDynamicRank()) {
    return {input_shape};
  }
  const auto &weight_shape = input_infos[idxes_.weight_idx]->GetShape();
  auto [batched_input_shape, is_batched] = Batchify(input_shape, 2, primitive->name());
  auto output_shape = ConvNdInferShape(primitive, input_infos, batched_input_shape, weight_shape, true);
  if (!is_batched) {
    output_shape.erase(output_shape.begin());
  }
  return {std::move(output_shape)};
}
}  // namespace ops
}  // namespace mindspore
