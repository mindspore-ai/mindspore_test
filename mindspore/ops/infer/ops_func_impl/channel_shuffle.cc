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

#include "infer/ops_func_impl/channel_shuffle.h"
#include <vector>
#include <string>
#include <set>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
ShapeArray ChannelShuffleFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();
  const auto &input = input_infos[kInputIndex0];
  MS_EXCEPTION_IF_NULL(input);
  auto input_shape = input->GetShape();
  const auto &groups_opt = input_infos[kInputIndex1]->GetScalarValue<int64_t>();

  if (!input->IsDynamic()) {
    constexpr int64_t kInputShapeSizeMin = 3;
    constexpr int64_t kInputShapeSizeMax = 7;
    auto input_rank = SizeToLong(input_shape.size());
    CheckAndConvertUtils::CheckInRange("rank of input shape", input_rank, kIncludeBoth,
                                       {kInputShapeSizeMin, kInputShapeSizeMax}, op_name);
    if (groups_opt.has_value()) {
      int64_t groups = groups_opt.value();
      int64_t c = input_shape[1];
      constexpr int64_t kZero = 0;
      CheckAndConvertUtils::CheckInteger("groups", groups, kGreaterThan, kZero, op_name);
      if ((c % groups) != kZero) {
        MS_EXCEPTION(ValueError) << "For ChannelShuffle, number of channels: " << c
                                 << ", must be divisible by groups: " << groups << ".";
      }
    }
  }
  return {input_shape};
}

std::vector<TypeId> ChannelShuffleFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const InferInfoPtrList &input_infos) const {
  const auto &prim_name = primitive->name();
  const auto input_type = input_infos[kInputIndex0]->GetType();
  const auto groups_type = input_infos[kInputIndex1]->GetType();
  CheckAndConvertUtils::CheckTypeIdValid("groups type", groups_type, common_integral_type_ids, prim_name);
  return {input_type};
}
}  // namespace ops
}  // namespace mindspore
