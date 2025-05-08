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

#include "infer/ops_func_impl/format_cast.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
int32_t FormatCastFuncImpl::CheckValidation(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto acl_format_opt = input_infos[kIndex1]->GetScalarValue<int64_t>();
  MS_CHECK_VALUE(acl_format_opt.has_value(), "For FormatCast, 'acl_format' should be a constant value.");
  return OP_CHECK_SUCCESS;
}

ShapeArray FormatCastFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetShape()};
}

std::vector<TypeId> FormatCastFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
