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

#include "infer/ops_func_impl/matmul_allreduce_add_rmsnorm.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {

ShapeArray MatmulAllReduceAddRmsNormFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                         const InferInfoPtrList &input_infos) const {
  ShapeArray out_shapes;
  auto &x1_infer_info = input_infos[kIndex0];
  auto &x2_infer_info = input_infos[kIndex1];
  auto &bias_infer_info = input_infos[kIndex2];
  auto &residual_infer_info = input_infos[kIndex3];
  auto &gamma_infer_info = input_infos[kIndex4];

  if (x1_infer_info->IsDynamicRank() || x2_infer_info->IsDynamicRank() || bias_infer_info->IsDynamicRank() ||
      residual_infer_info->IsDynamicRank() || gamma_infer_info->IsDynamicRank()) {
    return {{-2}, {-2}};
  }

  auto residual_shape = residual_infer_info->GetShape();
  out_shapes.emplace_back(residual_shape);
  out_shapes.emplace_back(residual_shape);
  return out_shapes;
}

TypeIdList MatmulAllReduceAddRmsNormFuncImpl::InferType(const PrimitivePtr &primitive,
                                                        const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kIndex3]->GetType();
  TypeIdList valid_types = {kNumberTypeFloat16, kNumberTypeBFloat16};
  bool is_supported_types =
    std::any_of(valid_types.begin(), valid_types.end(), [&](const TypeId &type) { return type == input_type; });
  if (!is_supported_types) {
    MS_EXCEPTION(TypeError) << "input only support float16 and bfloat16.";
  }
  return {input_type, input_type};
}

}  // namespace ops
}  // namespace mindspore
