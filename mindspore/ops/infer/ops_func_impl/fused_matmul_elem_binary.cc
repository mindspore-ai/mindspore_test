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

#include "infer/ops_func_impl/fused_matmul_elem_binary.h"
#include "infer/ops_func_impl/matmul_fusion_utils.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr FusedMatmulElemBinaryFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) const {
  const std::string binary_name = primitive->name();
  auto x = CheckAndConvertUtils::CheckArgsType(binary_name, input_args, kIndex0, kObjectTypeTensorType);
  auto y = CheckAndConvertUtils::CheckArgsType(binary_name, input_args, kIndex1, kObjectTypeTensorType);
  auto bias = CheckAndConvertUtils::CheckArgsType(binary_name, input_args, kIndex2, kObjectTypeTensorType);
  const auto input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input num", SizeToLong(input_args.size()), kEqual, input_num, binary_name);
  const auto &x_shp = x->GetShape()->GetShapeVector();
  const auto &y_shp = y->GetShape()->GetShapeVector();
  const auto &bias_shp = bias->GetShape()->GetShapeVector();
  (void)CheckAndConvertUtils::CheckInteger("bias rank", SizeToLong(bias_shp.size()), kEqual, 1, binary_name);
  bool transpose_a = primitive->HasAttr("is_trans_a") ? GetValue<bool>(primitive->GetAttr("is_trans_a")) : false;
  bool transpose_b = primitive->HasAttr("is_trans_b") ? GetValue<bool>(primitive->GetAttr("is_trans_b")) : false;
  return InferShape2D(x_shp, y_shp, transpose_a, transpose_b);
}

TypePtr FusedMatmulElemBinaryFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  const int binary_input_num = 3;
  return MatmulFusionUtils::FusedMatMulElemInferType(primitive, input_args, binary_input_num);
}

}  // namespace ops
}  // namespace mindspore
