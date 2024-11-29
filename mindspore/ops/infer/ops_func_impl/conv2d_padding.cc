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

#include "infer/ops_func_impl/conv2d_padding.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

#include "abstract/dshape.h"
#include "mindapi/base/types.h"
#include "mindspore/ops/op_def/op_name.h"
#include "utils/log_adapter.h"
#include "utils/shape_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kConv2dPaddingInputArgsSize = 7;
}  // namespace

ShapeArray Conv2DPaddingFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  if (input_infos.size() != kConv2dPaddingInputArgsSize) {
    MS_LOG(EXCEPTION) << "input args size should be  " << kConv2dPaddingInputArgsSize << ", but got "
                      << input_infos.size();
  }

  auto &input_tensor = input_infos[kIndex0];
  auto &weight_tensor = input_infos[kIndex1];
  auto input_shape = input_tensor->GetShape();
  auto weight_shape = weight_tensor->GetShape();
  auto prim_name = primitive->name();
  auto padding_opt = input_infos[kIndex4]->GetScalarValue<int64_t>();

  const auto rank_2d_isbatch = 4;
  const auto rank_2d_notbatch = 3;
  int64_t Co = abstract::Shape::kShapeDimAny;
  int64_t Ho = abstract::Shape::kShapeDimAny;
  int64_t Wo = abstract::Shape::kShapeDimAny;
  int64_t N = abstract::Shape::kShapeDimAny;
  if (input_tensor->IsDynamicRank() || weight_tensor->IsDynamicRank()) {
    if (MS_LIKELY(!input_tensor->IsDynamicRank())) {
      auto input_rank = SizeToLong(input_shape.size());
      MS_CHECK_VALUE(input_rank == rank_2d_isbatch || input_rank == rank_2d_notbatch,
                     CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>("rank of input", input_rank, kIncludeBoth,
                                                                          {3, 4}, primitive));
      N = input_shape[kIndex0];
      if (padding_opt.has_value()) {
        mindspore::PadMode padding_enum_value = static_cast<mindspore::PadMode>(padding_opt.value());
        if (padding_enum_value == PadMode::SAME) {
          Ho = input_shape[kIndex2];
          Wo = input_shape[kIndex3];
        }
      }
      auto output_shape = {N, Co, Ho, Wo};
      return {output_shape};
    }
    if (MS_LIKELY(!weight_tensor->IsDynamicRank())) {
      (void)CheckAndConvertUtils::CheckInteger("weight rank", SizeToLong(weight_shape.size()), kEqual, rank_2d_isbatch,
                                               prim_name);
      Co = weight_shape[kIndex0];
      auto output_shape = {N, Co, Ho, Wo};
      return {output_shape};
    }
    std::vector<int64_t> output_shape = {abstract::Shape::kShapeRankAny};
    return {output_shape};
  }

  auto input_rank = SizeToLong(input_shape.size());
  MS_CHECK_VALUE(
    input_rank == rank_2d_isbatch || input_rank == rank_2d_notbatch,
    CheckAndConvertUtils::FormatCheckInRangeMsg<int64_t>("rank of input", input_rank, kIncludeBoth, {3, 4}, primitive));
  int64_t weight_rank = SizeToLong(weight_shape.size());
  (void)CheckAndConvertUtils::CheckInteger("weight rank", weight_rank, kEqual, rank_2d_isbatch, prim_name);
  auto output_padding_2d = ShapeVector(weight_rank - 2, 0);
  return ConvNdInferShape(primitive, input_infos, input_shape, weight_shape, output_padding_2d, true);
}

std::vector<TypeId> Conv2DPaddingFuncImpl::InferType(const PrimitivePtr &primitive,
                                                     const InferInfoPtrList &input_infos) const {
  return {input_infos[kIndex0]->GetType()};
}
}  // namespace ops
}  // namespace mindspore
