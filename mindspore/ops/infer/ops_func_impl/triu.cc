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
#include "infer/ops_func_impl/triu.h"
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
ShapeArray TriuFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto &x = input_infos[kInputIndex0];
  ShapeVector x_shape = x->GetShape();
  if (MS_LIKELY(x->IsDynamic())) {
    return {x_shape};
  }
  auto input_shape_rank = SizeToLong(x_shape.size());
  const int64_t kMinShapeSize = 2;
  MS_CHECK_VALUE(input_shape_rank >= kMinShapeSize,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("rank of input", input_shape_rank, kGreaterEqual,
                                                             kMinShapeSize, primitive));
  return {x_shape};
}

std::vector<TypeId> TriuFuncImpl::InferType(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto type = input_infos[kInputIndex0]->GetType();
  return {type};
}
}  // namespace ops
}  // namespace mindspore
