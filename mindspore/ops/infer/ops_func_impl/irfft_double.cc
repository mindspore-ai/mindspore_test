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
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/irfft_double.h"

namespace mindspore {
namespace ops {
BaseShapePtr IRFFTDoubleFuncImpl::InferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr IRFFTDoubleFuncImpl::InferType(const PrimitivePtr &primitive,
                                       const std::vector<AbstractBasePtr> &input_args) const {
  MS_EXCEPTION_IF_NULL(input_args[0]);
  MS_EXCEPTION_IF_NULL(input_args[0]->GetType());
  return input_args[0]->GetType()->Clone();
}

/*
Error list:
1) `input.ndim` is not in the range of "[1, 8]".
2) The value in `dim` is not in the range of "[-`input.ndim`, `input.ndim`)"
3) The value in `n` is less than or equal to 0.
*/
int32_t IRFFTDoubleFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  auto check_status = OP_CHECK_SUCCESS;
  const auto &input_x_shape = input_args[kIndex0]->GetShape();
  auto x_shape_vec = input_x_shape->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec))) {
    check_status = OP_CHECK_RETRY;
  }
  const int64_t x_min_rank = 1;
  const int64_t x_max_rank = 8;
  int64_t x_rank = SizeToLong(x_shape_vec.size());

  MS_CHECK_VALUE(x_shape_vec.size() >= x_min_rank && x_shape_vec.size() <= x_max_rank,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("rank of input", x_rank, kIncludeBoth,
                                                             {x_min_rank, x_max_rank}, primitive));

  if (x_rank == 1 && x_shape_vec[0] == 0) {
    MS_EXCEPTION(ValueError) << "Unsupported input shape dimension. The shape should not be empty.";
  }

  if (!input_args[kInputIndex1]->GetType()->isa<TypeNone>()) {
    auto n_opt = GetScalarValue<int64_t>(input_args[kInputIndex1]->GetValue());
    if (n_opt.has_value()) {
      int64_t n = n_opt.value();
      (void)CheckAndConvertUtils::CheckInteger("n", n, kGreaterThan, 0);
    }
  }

  auto dim_opt = GetScalarValue<int64_t>(input_args[kInputIndex2]->GetValue());
  if (dim_opt.has_value()) {
    int64_t dim = dim_opt.value();
    MS_CHECK_VALUE(dim >= -x_rank && dim < x_rank, CheckAndConvertUtils::FormatCheckInRangeMsg(
                                                     "dim", dim, kIncludeRight, {-x_rank, x_rank}, primitive));
  }

  return check_status;
}
}  // namespace ops
}  // namespace mindspore
