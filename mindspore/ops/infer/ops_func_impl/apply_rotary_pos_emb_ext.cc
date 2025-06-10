/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/apply_rotary_pos_emb_ext.h"

#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/helper.h"

namespace mindspore {
namespace ops {
int32_t ApplyRotaryPosEmbExtFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  auto &query = input_infos[kIndex0];
  auto &key = input_infos[kIndex1];
  auto &cos = input_infos[kIndex2];
  auto &sin = input_infos[kIndex3];

  MS_CHECK_VALUE(query->GetShape().size() == 4, "For ApplyRotaryPosEmbExt, Query must be a 4D tensor, but got shape " +
                                                  ShapeVectorToStr(query->GetShape()));
  MS_CHECK_VALUE(key->GetShape().size() == 4, "For ApplyRotaryPosEmbExt, key must be a 4D tensor, but got shape " +
                                                ShapeVectorToStr(key->GetShape()));
  MS_CHECK_VALUE(cos->GetShape().size() == 4, "For ApplyRotaryPosEmbExt, cos must be a 4D tensor, but got shape " +
                                                ShapeVectorToStr(cos->GetShape()));
  MS_CHECK_VALUE(sin->GetShape().size() == 4, "For ApplyRotaryPosEmbExt, sin must be a 4D tensor, but got shape " +
                                                ShapeVectorToStr(sin->GetShape()));
  return OP_CHECK_SUCCESS;
}

ShapeArray ApplyRotaryPosEmbExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_infos[kIndex0]->IsDynamicRank() || input_infos[kIndex1]->IsDynamicRank()) {
    return {ShapeVector{abstract::TensorShape::kShapeRankAny}};
  }
  return {input_infos[kIndex0]->GetShape(), input_infos[kIndex1]->GetShape()};
}

std::vector<TypeId> ApplyRotaryPosEmbExtFuncImpl::InferType(const PrimitivePtr &prim,
                                                            const InferInfoPtrList &input_infos) const {
  auto query_dtype = input_infos[kIndex0]->GetType();
  auto key_dtype = input_infos[kIndex1]->GetType();
  return {query_dtype, key_dtype};
}
}  // namespace ops
}  // namespace mindspore
