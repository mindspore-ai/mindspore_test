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

#include "mindspore/ops/infer/ops_func_impl/all_gather_matmul.h"

#include <set>
#include <string>
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
ShapeArray AllGatherMatmulFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();

  auto &input_tensor = input_infos[kAllGatherMatmulInputInputIndex];
  auto &x2_tensor = input_infos[kAllGatherMatmulInputX2Index];
  auto trans_input_optional = input_infos[kAllGatherMatmulInputTransInputIndex]->GetScalarValue<bool>();
  auto trans_x2_optional = input_infos[kAllGatherMatmulInputTransX2Index]->GetScalarValue<bool>();
  auto world_size_optional = input_infos[kAllGatherMatmulInputWorldSizeIndex]->GetScalarValue<int64_t>();
  auto gather_output_optional = input_infos[kAllGatherMatmulInputGatherOutputIndex]->GetScalarValue<bool>();

  auto input_shape = input_tensor->GetShape();
  auto x2_shape = x2_tensor->GetShape();
  auto input_row = 0;
  auto input_col = 1;
  auto x2_row = 0;
  auto x2_col = 1;
  size_t supported_rank = 2;
  if (trans_input_optional.has_value() && trans_input_optional.value()) {
    input_row = 1;
    input_col = 0;
  }
  if (trans_x2_optional.has_value() && trans_x2_optional.value()) {
    x2_row = 1;
    x2_col = 0;
  }
  CheckRank(input_tensor, supported_rank, op_name, "input");
  CheckRank(x2_tensor, supported_rank, op_name, "x2");
  auto input_row_known = trans_input_optional.has_value() && IsShapeKnown(input_tensor, input_row);
  auto input_col_known = trans_input_optional.has_value() && IsShapeKnown(input_tensor, input_col);
  auto x2_row_known = trans_x2_optional.has_value() && IsShapeKnown(x2_tensor, x2_row);
  auto x2_col_known = trans_x2_optional.has_value() && IsShapeKnown(x2_tensor, x2_col);

  if (input_col_known && x2_row_known && input_shape[input_col] != x2_shape[x2_row]) {
    MS_LOG(EXCEPTION) << op_name << ": The column of input and the row of x2 must be equal, but the column of input is "
                      << input_shape[input_col] << " and the row of x2 is " << x2_shape[x2_row];
  }

  MS_ASSERT(world_size_optional.has_value());
  auto world_size = world_size_optional.value();

  static constexpr ShapeValueDType kShapeRankAny = mindspore::abstract::Shape::kShapeRankAny;
  static constexpr ShapeValueDType kShapeDimAny = mindspore::abstract::Shape::kShapeDimAny;

  ShapeVector output_shape(2);  // The rank of output is 2.
  output_shape[0] = input_row_known ? input_shape[input_row] * world_size : kShapeDimAny;
  output_shape[1] = x2_col_known ? x2_shape[x2_col] : kShapeDimAny;

  ShapeVector gather_out_shape = {kShapeRankAny};
  if (gather_output_optional.has_value()) {
    gather_out_shape = {0};
    if (gather_output_optional.value()) {
      gather_out_shape = {kShapeDimAny, kShapeDimAny};
      if (input_row_known) {
        gather_out_shape[0] = input_shape[input_row] * world_size;
      }
      if (input_col_known) {
        gather_out_shape[1] = input_shape[input_col];
      } else if (x2_row_known) {
        gather_out_shape[1] = x2_shape[x2_row];
      }
    }
  }

  return {output_shape, gather_out_shape};
}

std::vector<TypeId> AllGatherMatmulFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  auto op_name = primitive->name();

  auto input_type = input_infos[kAllGatherMatmulInputInputIndex]->GetType();
  auto x2_type = input_infos[kAllGatherMatmulInputX2Index]->GetType();
  if (input_type != x2_type) {
    MS_LOG(EXCEPTION) << op_name << ": the dtype of input and the dtype of x2 must be the same, "
                      << "but the dtype of input is " << input_type << " and the dtype of x2 is " << x2_type;
  }
  if (!input_infos[kAllGatherMatmulInputBiasIndex]->IsNone()) {
    MS_LOG(EXCEPTION) << op_name << ": bias must be None.";
  }

  // Set group attribute to primitive
  if (!primitive->HasAttr(kAttrGroup)) {
    auto group = input_infos[kAllGatherMatmulInputGroupIndex]->GetScalarValue<std::string>();
    if (group.has_value()) {
      (void)primitive->AddAttr(kAttrGroup, MakeValue(group.value()));
    }
  }

  return {input_type, input_type};
}
}  // namespace ops
}  // namespace mindspore
