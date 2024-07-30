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

#include "infer/ops_func_impl/quantbatchmatmul_split_out3.h"
#include "infer/ops_func_impl/matmul_fusion_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {

BaseShapePtr QuantbatchmatmulSplitOut3FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) const {
  return MatmulFusionUtils::InferenceMultiMatmulInferShape(primitive, input_args);
}

TypePtr QuantbatchmatmulSplitOut3FuncImpl::InferType(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  return MatmulFusionUtils::InferenceMultiMatmulInferType(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
