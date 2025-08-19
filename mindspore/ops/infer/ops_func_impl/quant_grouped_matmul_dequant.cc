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

#include "infer/ops_func_impl/quant_grouped_matmul_dequant.h"
#include <memory>
#include <set>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr size_t kXDim = 2;
constexpr size_t kWeightNDFormatDim = 3;
ShapeArray QuantGroupedMatmulDequantFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                         const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  ShapeVector x_shape = input_infos[kInputIndex0]->GetShape();
  ShapeVector weight_shape = input_infos[kInputIndex1]->GetShape();

  if (x_shape.size() != kXDim) {
    MS_LOG(EXCEPTION) << "Input x must be 2D, but got " << x_shape.size();
  }

  if (weight_shape.size() != kWeightNDFormatDim) {
    MS_LOG(EXCEPTION) << "Input weight must be 3D, but got " << weight_shape.size();
  }

  if (MS_UNLIKELY(IsDynamicRank(x_shape) || IsDynamicRank(weight_shape))) {
    ShapeVector out_shape = {abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny};
    return {out_shape};
  }

  bool transpose_weight = true;
  if (primitive->HasAttr("transpose_weight")) {
    transpose_weight = GetValue<bool>(primitive->GetAttr("transpose_weight"));
  }

  int64_t m = x_shape[kIndex0];
  int64_t n = transpose_weight ? weight_shape[kIndex1] : weight_shape[kIndex2];
  ShapeVector out_shape = {m, n};
  return {out_shape};
}

std::vector<TypeId> QuantGroupedMatmulDequantFuncImpl::InferType(const PrimitivePtr &primitive,
                                                                 const InferInfoPtrList &input_infos) const {
  return {kNumberTypeFloat16};
}
}  // namespace ops
}  // namespace mindspore
