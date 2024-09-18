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

#include "infer/ops_func_impl/fft_shapecopy.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ccsrc/include/common/utils/utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr FFTShapeCopyFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  auto dout_shape_ptr = input_args[kIndex0]->GetShape();
  auto dout_shape = dout_shape_ptr->GetShapeVector();
  if (IsDynamicRank(dout_shape)) {
    ShapeVector dyn_output{abstract::TensorShape::kShapeRankAny};
    return std::make_shared<abstract::TensorShape>(dyn_output);
  }

  auto x_shape = dout_shape;
  auto shape_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]->GetValue());
  if (shape_opt.has_value()) {
    std::vector<int64_t> shape = shape_opt.value().ToVector();
    for (size_t i = 0; i < shape.size(); i++) {
      x_shape[i] = shape[i];
    }
  }

  return std::make_shared<abstract::TensorShape>(x_shape);
}

TypePtr FFTShapeCopyFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType()->Clone();
}

/*
  Error list:
  1) `input.ndim` is not in the range of "[1, 8]".
  2) The value in `shape` is less than or equal to 0.
*/
int32_t FFTShapeCopyFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  auto check_status = OP_CHECK_SUCCESS;
  const auto &input_x_shape = input_args[kIndex0]->GetShape();
  auto x_shape_vec = input_x_shape->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec))) {
    check_status = OP_CHECK_RETRY;
  }

  const int64_t kMinRank = 1;
  const int64_t kMaxRank = 8;
  int64_t x_rank = SizeToLong(x_shape_vec.size());

  if (x_shape_vec.size() < kMinRank || x_shape_vec.size() > kMaxRank) {
    MS_EXCEPTION(ValueError) << CheckAndConvertUtils::FormatCheckInRangeMsg("rank of input", x_rank, kIncludeBoth,
                                                                            {kMinRank, kMaxRank}, primitive);
  }

  if (std::accumulate(x_shape_vec.begin(), x_shape_vec.end(), 0) == 0) {
    MS_EXCEPTION(ValueError) << "Unsupported input shape dimension. The shape should not be empty.";
  }

  auto shape_opt = GetArrayValue<int64_t>(input_args[kInputIndex1]->GetValue());
  if (shape_opt.has_value()) {
    std::vector<int64_t> shape = shape_opt.value().ToVector();
    for (size_t i = 0; i < shape.size(); i++) {
      (void)CheckAndConvertUtils::CheckInteger("shape", shape[i], kGreaterThan, 0);
    }
  }
  return check_status;
}
}  // namespace ops
}  // namespace mindspore
