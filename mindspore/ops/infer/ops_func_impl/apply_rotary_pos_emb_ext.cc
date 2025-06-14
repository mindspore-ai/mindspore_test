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

  MS_CHECK_VALUE(
    query->GetShape().size() == kApplyRotaryPosEmbExtShapeSize,
    "For ApplyRotaryPosEmbExt, Query must be a 4D tensor, but got shape " + ShapeVectorToStr(query->GetShape()));
  MS_CHECK_VALUE(
    key->GetShape().size() == kApplyRotaryPosEmbExtShapeSize,
    "For ApplyRotaryPosEmbExt, key must be a 4D tensor, but got shape " + ShapeVectorToStr(key->GetShape()));
  MS_CHECK_VALUE(
    cos->GetShape().size() == kApplyRotaryPosEmbExtShapeSize,
    "For ApplyRotaryPosEmbExt, cos must be a 4D tensor, but got shape " + ShapeVectorToStr(cos->GetShape()));
  MS_CHECK_VALUE(
    sin->GetShape().size() == kApplyRotaryPosEmbExtShapeSize,
    "For ApplyRotaryPosEmbExt, sin must be a 4D tensor, but got shape " + ShapeVectorToStr(sin->GetShape()));
  return OP_CHECK_SUCCESS;
}

ShapeArray ApplyRotaryPosEmbExtFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  if (input_infos[kIndex0]->IsDynamicRank() || input_infos[kIndex1]->IsDynamicRank() ||
      input_infos[kIndex2]->IsDynamicRank() || input_infos[kIndex3]->IsDynamicRank()) {
    return {input_infos[kIndex0]->GetShape(), input_infos[kIndex1]->GetShape()};
  }
  CheckValidation(primitive, input_infos);
  return {input_infos[kIndex0]->GetShape(), input_infos[kIndex1]->GetShape()};
}

std::vector<TypeId> ApplyRotaryPosEmbExtFuncImpl::InferType(const PrimitivePtr &primitive,
                                                            const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto query_dtype = input_infos[kIndex0]->GetType();
  auto key_dtype = input_infos[kIndex1]->GetType();
  auto cos_dtype = input_infos[kIndex2]->GetType();
  auto sin_dtype = input_infos[kIndex3]->GetType();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const auto &soc_version = ms_context->ascend_soc_version();

  if (soc_version == kAscendVersion910_93 || soc_version == kAscendVersion910b) {
    const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32};
    CheckAndConvertUtils::CheckTypeIdValid("query", query_dtype, valid_types, op_name);
    CheckAndConvertUtils::CheckTypeIdValid("key", key_dtype, valid_types, op_name);
    CheckAndConvertUtils::CheckTypeIdValid("cos", cos_dtype, valid_types, op_name);
    CheckAndConvertUtils::CheckTypeIdValid("sin", sin_dtype, valid_types, op_name);
  } else if (soc_version == kAscendVersion310p) {
    const std::set<TypeId> valid_types = {kNumberTypeFloat16, kNumberTypeFloat32};
    CheckAndConvertUtils::CheckTypeIdValid("query", query_dtype, valid_types, op_name);
    CheckAndConvertUtils::CheckTypeIdValid("key", key_dtype, valid_types, op_name);
    CheckAndConvertUtils::CheckTypeIdValid("cos", cos_dtype, valid_types, op_name);
    CheckAndConvertUtils::CheckTypeIdValid("sin", sin_dtype, valid_types, op_name);
  } else {
    MS_LOG(EXCEPTION) << "'ApplyRotaryPosEmbExt' only support [" << kAscendVersion910b << ", " << kAscendVersion910_93
                      << ", " << kAscendVersion310p << "], but got " << soc_version;
  }
  return {query_dtype, key_dtype};
}
}  // namespace ops
}  // namespace mindspore
