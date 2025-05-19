/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/masked_fill.h"
#include "ops_utils/op_constants.h"

namespace mindspore::ops {
ShapeArray MaskedFillFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto input_shape = input_infos[kIndex0]->GetShape();
  auto mask_shape = input_infos[kIndex1]->GetShape();
  auto value_shape = input_infos[kIndex2]->GetShape();
  auto broadcast_shape = CalBroadCastShape(input_shape, mask_shape, op_name, "input", "mask");
  int64_t batch_rank = 0;

  if (primitive->HasAttr(kBatchRank)) {
    auto value_ptr = primitive->GetAttr(kBatchRank);
    batch_rank = GetValue<int64_t>(value_ptr);
  }
  if (batch_rank == 0 && value_shape.size() != 0) {
    MS_EXCEPTION(ValueError)
      << "For '" << op_name
      << "', 'value' only supports a 0-dimensional value tensor or a float number, but got tensor with "
      << value_shape.size() << " dimension(s).";
  } else if (value_shape.size() != 0) {
    (void)CheckAndConvertUtils::CheckInteger("value shape size", SizeToLong(value_shape.size()), kEqual, batch_rank,
                                             op_name);
    (void)CheckAndConvertUtils::CheckInteger("value shape size", SizeToLong(value_shape.size()), kLessEqual,
                                             SizeToLong(broadcast_shape.size()), op_name);
    for (size_t i = 0; i < LongToSize(batch_rank); i++) {
      if (value_shape[i] != broadcast_shape[i]) {
        MS_EXCEPTION(ValueError) << "For '" << op_name << "', the " << i
                                 << "th index of value shape should be equal to " << broadcast_shape[i] << ", but got "
                                 << value_shape[i];
      }
    }
  }

  return {broadcast_shape};
}

std::vector<TypeId> MaskedFillFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  auto mask_type_id = input_infos[kIndex1]->GetType();
  (void)CheckAndConvertUtils::CheckTypeValid("mask", TypeIdToType(mask_type_id), {kBool}, op_name);
  return {input_infos[kInputIndex0]->GetType()};
}
}  // namespace mindspore::ops
