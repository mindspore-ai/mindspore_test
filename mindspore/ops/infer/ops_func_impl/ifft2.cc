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

#include <set>
#include <memory>
#include <unordered_map>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "infer/ops_func_impl/fft_arithmetic.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/ifft2.h"

namespace mindspore {
namespace ops {
BaseShapePtr IFFT2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  return FFTNInferShape(primitive, input_args);
}

TypePtr IFFT2FuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return FFTInferType(primitive, input_args);
}

/*
  Error list:
  1) `input.ndim` is not in the range of "[1, 8]".
  2) The value in `dim` is not in the range of "[-`input.ndim`, `input.ndim`)"
  3) The value in `n` is less than or equal to 0.
*/
int32_t IFFT2FuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  return FFTNCheckValidation(primitive, input_args);
}
}  // namespace ops
}  // namespace mindspore
