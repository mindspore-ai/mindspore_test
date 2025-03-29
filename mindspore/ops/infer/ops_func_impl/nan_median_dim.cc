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

#include <vector>
#include "infer/ops_func_impl/nan_median_dim.h"
#include "op_def/auto_generate/gen_ops_name.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "mindspore/ops/op_def/op_name.h"

namespace mindspore {
namespace ops {
ShapeArray NanMedianDimFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const auto &input = input_infos[kInputIndex0];
  const auto keepdim_opt = input_infos[kInputIndex2]->GetScalarValue<bool>();
  // 1. Input is dynamic rank -> dynamic rank.
  if (input->IsDynamicRank()) {
    return TwiceOf({abstract::Shape::kShapeRankAny});
  }
  const auto &input_shape = input->GetShape();
  // If input is scalar, keepdim has no effect. Only check dim.
  if (MS_UNLIKELY(input_shape.empty())) {
    return ScalarCheckDim(input_infos);
  }
  // For Tensor, check and validate dim and keepdim.
  // 2. Keepdim is unknown -> dynamic rank.
  if (!keepdim_opt.has_value()) {
    return TwiceOf({abstract::Shape::kShapeRankAny});
  }
  const auto keepdim = keepdim_opt.value();
  // 3. Dim is unknown -> dynamic shape on all dim.
  const auto input_rank = input_shape.size();
  const auto &dim_opt = input_infos[kInputIndex1]->GetScalarValue<int64_t>();
  if (!dim_opt.has_value()) {
    return TwiceOf(ShapeAnyOf(keepdim ? input_rank : input_rank - 1));
  }
  // 4. Dim and keepdim are known, check dim in range.
  const auto raw_dim = dim_opt.value();
  const auto dim = CheckDim(raw_dim, input_rank);
  // 5. Check if input.shape[dim] is empty
  if (MS_UNLIKELY(input_shape[dim] == 0)) {
    MS_EXCEPTION(IndexError) << "For nanmedian, reduction dimension " << raw_dim
                             << " should have non-zero size, but got input.shape[" << raw_dim << "] == 0.";
  }
  auto out_shape = input_shape;
  if (keepdim) {
    out_shape[dim] = 1;
  } else {
    out_shape.erase(out_shape.begin() + dim);
  }
  return TwiceOf(out_shape);
}

std::vector<TypeId> NanMedianDimFuncImpl::InferType(const PrimitivePtr &primitive,
                                                    const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType(), TypeId::kNumberTypeInt64};
}

int64_t NanMedianDimFuncImpl::CheckDim(int64_t dim, size_t input_rank) {
  const auto dim_range = SizeToLong(input_rank);
  if (MS_UNLIKELY((dim < -dim_range) || (dim >= dim_range))) {
    MS_EXCEPTION(IndexError) << "For nanmedian, dimension should be in range of [" << -dim_range << ", "
                             << (dim_range - 1) << "], but got " << dim << ".";
  }
  return dim < 0 ? dim + dim_range : dim;
}

ShapeArray NanMedianDimFuncImpl::ScalarCheckDim(const InferInfoPtrList &input_infos) {
  const auto &dim_opt = input_infos[kInputIndex1]->GetScalarValue<int64_t>();
  if (!dim_opt.has_value()) {
    return TwiceOf({abstract::Shape::kShapeDimAny});
  }
  const auto raw_dim = dim_opt.value();
  (void)CheckDim(raw_dim, 1);  // rank = 1 to check in range [-1, 0]
  return TwiceOf({});
}
}  // namespace ops
}  // namespace mindspore
