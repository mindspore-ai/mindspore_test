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

#include "infer/ops_func_impl/reverse_v2.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"
#include "ops_utils/op_constants.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace ops {
int32_t ReverseV2FuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto &x_shape_vec = input_args[kIndex0]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamicRank(x_shape_vec))) {
    return OP_CHECK_RETRY;
  }

  constexpr int64_t kMaxRank = 8;
  const int64_t x_rank = SizeToLong(x_shape_vec.size());
  MS_CHECK_VALUE(x_rank != 0,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("rank of x", x_rank, kNotEqual, 0, primitive));
  MS_CHECK_VALUE(x_rank <= kMaxRank,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("rank of x", x_rank, kLessEqual, kMaxRank, primitive));

  const auto &axis_opt = GetArrayValue<int64_t>(input_args[kIndex1]);
  if (MS_UNLIKELY(!axis_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  const auto &axis_array = axis_opt.value();
  if (MS_UNLIKELY(axis_array.HasUnknownValue())) {
    return OP_CHECK_RETRY;
  }

  constexpr uint32_t kBitOne = 1;
  uint32_t axis_bitmap = 0;
  const auto &axis = axis_array.ToVector();
  for (size_t i = 0; i < axis.size(); ++i) {
    auto real_dim = axis[i] < 0 ? axis[i] + x_rank : axis[i];
    if (real_dim < 0 || real_dim >= x_rank) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the 'axis[" << i << "]' must be in range of [-"
                               << x_rank << ", " << x_rank << "), but got " << axis[i] << " with type 'int'.";
    } else if ((axis_bitmap >> real_dim) & kBitOne) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", 'axis' cannot contain duplicate dimensions"
                               << ", but got " << real_dim;
    } else {
      axis_bitmap |= (kBitOne << real_dim);
    }
  }
  return OP_CHECK_SUCCESS;
}

BaseShapePtr ReverseV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetShape()->Clone();
}

TypePtr ReverseV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}
TypePtrList ReverseV2FuncImpl::InferType(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  return {x_tensor->Dtype()};
}
ShapeArray ReverseV2FuncImpl::InferShape(const PrimitivePtr &primitive, const ValuePtrList &input_values) const {
  const auto &x_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(x_tensor);
  const auto &x_shape_vec = x_tensor->shape();
  const int64_t x_rank = SizeToLong(x_shape_vec.size());
  MS_CHECK_VALUE(x_rank != 0,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("rank of x", x_rank, kNotEqual, 0, primitive));
  const ArrayValue<int64_t> axis_array = GetArrayValue<int64_t>(input_values[kIndex1]).value();
  constexpr uint32_t kBitOne = 1;
  uint32_t axis_bitmap = 0;
  const std::vector<int64_t> axis = axis_array.ToVector();
  for (size_t i = 0; i < axis.size(); ++i) {
    auto real_dim = axis[i] < 0 ? axis[i] + x_rank : axis[i];
    if (real_dim < 0 || real_dim >= x_rank) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the 'axis[" << i << "]' must be in range of [-"
                               << x_rank << ", " << x_rank << "), but got " << axis[i] << " with type 'int'.";
    } else if ((axis_bitmap >> real_dim) & kBitOne) {
      MS_EXCEPTION(ValueError) << "For " << primitive->name() << ", 'axis' cannot contain duplicate dimensions"
                               << ", but got " << real_dim;
    } else {
      axis_bitmap |= (kBitOne << real_dim);
    }
  }
  return {x_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameReverseV2, ReverseV2FuncImpl)
}  // namespace ops
}  // namespace mindspore
