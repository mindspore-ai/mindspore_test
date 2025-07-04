/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <memory>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/irfft.h"
#include "infer/ops_func_impl/fft_arithmetic.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
BaseShapePtr IRFFTFuncImpl::InferShape(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  return FFTInferShape(primitive, input_args);
}

TypePtr IRFFTFuncImpl::InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  return FFTInferType(primitive, input_args);
}

/*
Error list:
1) `input.ndim` is not in the range of "[1, 8]".
2) The value in `dim` is not in the range of "[-`input.ndim`, `input.ndim`)"
3) The value in `n` is less than or equal to 0.
*/
int32_t IRFFTFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  return FFTCheckValidation(primitive, input_args);
}

}  // namespace ops
}  // namespace mindspore
