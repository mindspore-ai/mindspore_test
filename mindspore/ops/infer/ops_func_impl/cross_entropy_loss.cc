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

#include "infer/ops_func_impl/cross_entropy_loss.h"
#include <string>
#include <set>
#include <algorithm>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kValidDim = 2;
int64_t CrossEntropyLossCheckDimSame(const int64_t dim1, const int64_t dim2, const char *dim_name) {
  if (dim1 != abstract::Shape::kShapeDimAny && dim2 != abstract::Shape::kShapeDimAny && dim1 != dim2) {
    MS_EXCEPTION(ValueError) << "For 'CrossEntropyLoss', " << dim_name << "-dim of inputs should be the same, but got "
                             << dim1 << " and " << dim2 << " respectively.";
  }
  return dim2 == abstract::Shape::kShapeDimAny ? dim1 : dim2;
}

ShapeVector CrossEntropyLossGetShapeInfo(const ShapeVector &input_shape, const ShapeVector &target_shape,
                                         const ShapeVector &weight_shape) {
  auto N_0 = (!IsDynamicRank(input_shape)) ? input_shape[0] : abstract::TensorShape::kShapeDimAny;
  auto N_1 = (!IsDynamicRank(target_shape)) ? target_shape[0] : abstract::TensorShape::kShapeDimAny;
  auto N = CrossEntropyLossCheckDimSame(N_0, N_1, "N");
  auto C_0 = (!IsDynamicRank(input_shape)) ? input_shape[1] : abstract::TensorShape::kShapeDimAny;
  auto C_2 = (!IsDynamicRank(weight_shape)) ? weight_shape[0] : abstract::TensorShape::kShapeDimAny;
  auto C = CrossEntropyLossCheckDimSame(C_0, C_2, "C");
  return std::vector<int64_t>{N, C};
}
}  // namespace

ShapeArray CrossEntropyLossFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const InferInfoPtrList &input_infos) const {
  auto &input = input_infos[kIndex0];
  auto &target = input_infos[kIndex1];
  auto &weight = input_infos[kIndex2];
  ShapeVector input_shape = input->GetShape();
  ShapeVector target_shape = target->GetShape();
  ShapeVector weight_shape = !weight->IsNone() ? weight->GetShape() : ShapeVector{abstract::Shape::kShapeDimAny};
  if (!IsDynamicRank(input_shape) && input_shape.size() != kValidDim) {
    MS_EXCEPTION(ValueError) << "For 'CrossEntropyLoss', 'input' must be 2-D Tensor, but got " << input_shape.size()
                             << "-D Tensor.";
  }
  auto shape_info = CrossEntropyLossGetShapeInfo(input_shape, target_shape, weight_shape);
  auto N = shape_info[0];
  auto C = shape_info[1];
  ShapeVector out_loss_shape = {N};
  ShapeVector out_log_prob_shape = {N, C};
  ShapeVector out_zloss_shape = {N};
  ShapeVector out_lse_for_zloss_shape = {N};
  const auto &reduction_opt = input_infos[kInputIndex3]->GetScalarValue<int64_t>();
  if (!reduction_opt.has_value()) {
    out_loss_shape = ShapeVector({abstract::TensorShape::kShapeRankAny});
  } else {
    auto reduction = static_cast<Reduction>(reduction_opt.value());
    if (reduction != Reduction::NONE) {
      out_loss_shape = ShapeVector({1});
    }
  }
  const auto &return_zloss_opt = input_infos[kInputIndex7]->GetScalarValue<bool>();
  if (!return_zloss_opt.has_value()) {
    out_zloss_shape = ShapeVector({abstract::TensorShape::kShapeDimAny});
  } else {
    if (return_zloss_opt.value() == false) {
      out_zloss_shape = {0};
    }
  }

  const auto &lse_for_zloss_opt = input_infos[kInputIndex6]->GetScalarValue<float>();
  if (!lse_for_zloss_opt.has_value()) {
    out_lse_for_zloss_shape = ShapeVector({abstract::TensorShape::kShapeDimAny});
  } else {
    if (lse_for_zloss_opt.value() == 0.0) {
      out_lse_for_zloss_shape = {0};
    }
  }

  return {out_loss_shape, out_log_prob_shape, out_zloss_shape, out_lse_for_zloss_shape};
}

std::vector<TypeId> CrossEntropyLossFuncImpl::InferType(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  auto type = input_infos[kInputIndex0]->GetType();
  return {type, type, type, type};
}
}  // namespace ops
}  // namespace mindspore
