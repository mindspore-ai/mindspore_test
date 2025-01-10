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
#include "infer/ops_func_impl/inplace_scatter_add.h"
#include <memory>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
ShapeArray InplaceScatterAddFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  const auto &prim_name = primitive->name();
  auto &input = input_infos[kInputIndex0];
  auto &index = input_infos[kInputIndex2];
  auto &src = input_infos[kInputIndex3];
  auto dim_opt = input_infos[kInputIndex1]->GetScalarValue<int64_t>();
  ShapeVector input_shape = input->GetShape();
  ShapeVector index_shape = index->GetShape();
  ShapeVector src_shape = src->GetShape();
  if (MS_LIKELY(input->IsDynamic()) || MS_LIKELY(index->IsDynamic()) || MS_LIKELY(src->IsDynamic()) ||
      MS_UNLIKELY(!dim_opt.has_value())) {
    return {input_shape};
  }
  auto dim = dim_opt.value();
  auto input_rank = SizeToLong(input_shape.size());
  MS_CHECK_VALUE(
    dim >= -input_rank && dim < input_rank,
    CheckAndConvertUtils::FormatCheckInRangeMsg("dim", dim, kIncludeLeft, {-input_rank, input_rank}, primitive));
  if (input_shape.size() < 1 || index_shape.size() < 1 || src_shape.size() < 1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", dimension size of 'input', 'index' and "
                             << "'src' must be greater than or equal to 1. But got input_shape: " << input_shape
                             << ", index_shape: " << index_shape << ", src_shape: " << src_shape << ".";
  }
  if (input_shape.size() != index_shape.size() || input_shape.size() != src_shape.size()) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name
                             << "', the dimension of 'input', 'index' and 'src' should be same, but got "
                             << "'input' dims: " << input_shape.size() << "; "
                             << "'index' dims: " << index_shape.size() << "; "
                             << "'src' dims: " << src_shape.size() << ".";
  }
  auto final_dim = dim >= 0 ? dim : dim + input_rank;
  for (int64_t d = 0; d < input_rank; d++) {
    if (d != final_dim && index_shape[d] > input_shape[d]) {
      MS_EXCEPTION(ValueError) << "For " << prim_name << ", "
                               << "except for the dimension specified by 'dim', the size of each dimension of 'index' "
                                  "must be less than or equal to that of 'input'. But got "
                               << d << "th dim of 'index' and 'input' " << index_shape[d] << ", " << input_shape[d]
                               << "respectively.";
    }
  }
  return {input_shape};
}

std::vector<TypeId> InplaceScatterAddFuncImpl::InferType(const PrimitivePtr &primitive,
                                                         const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kInputIndex0]->GetType();
  return {input_type};
}
}  // namespace ops
}  // namespace mindspore
