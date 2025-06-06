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

#include <memory>
#include "infer/ops_func_impl/avg_pool1d.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {

static inline int64_t AvgPool1DOutputShapePadLR(int64_t input_size, int64_t kernel_size, int64_t stride, int64_t pad,
                                                bool ceil_mode) {
  auto output_size = (input_size + 2 * pad - kernel_size + (ceil_mode ? stride - 1 : 0)) / stride + 1;
  // ensure that the last pooling starts inside the image needed to avoid problems in ceil mode
  if (ceil_mode && ((output_size - 1) * stride >= input_size + pad)) {
    --output_size;
  }
  return output_size;
}

ShapeArray AvgPool1DFuncImpl::InferShape(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos) const {
  const InferInfoPtr &x = input_infos[kInputIndex0];
  auto input_shape = x->GetShape();

  if (x->IsDynamicRank()) {
    return {input_shape};
  }

  auto input_rank = input_shape.size();
  auto last_dim_value = input_shape[input_rank - 1];
  input_shape[input_rank - 1] = abstract::TensorShape::kShapeDimAny;

  auto kernel_size_opt = input_infos[kInputIndex1]->GetArrayValue<int64_t>();
  std::optional<ArrayValue<int64_t>> stride_opt;
  if (input_infos[kInputIndex2]->IsNone()) {
    stride_opt = kernel_size_opt;
  } else {
    stride_opt = input_infos[kInputIndex2]->GetArrayValue<int64_t>();
  }

  auto padding_opt = input_infos[kInputIndex3]->GetArrayValue<int64_t>();
  auto ceil_mode_opt = input_infos[kInputIndex4]->GetScalarValue<bool>();
  if (MS_UNLIKELY(!(ceil_mode_opt.has_value() && kernel_size_opt.has_value() && stride_opt.has_value() &&
                    padding_opt.has_value()))) {
    return {input_shape};
  }

  auto kernel_size = kernel_size_opt.value();
  auto stride = stride_opt.value();
  auto padding = padding_opt.value();
  if (MS_UNLIKELY(last_dim_value == abstract::TensorShape::kShapeDimAny || kernel_size.IsValueUnknown(0) ||
                  stride.IsValueUnknown(0) || padding.IsValueUnknown(0))) {
    return {input_shape};
  }
  auto kernel_size_val = kernel_size[0];
  auto stride_val = stride[0];
  auto padding_val = padding[0];

  if (kernel_size_val < 1) {
    MS_EXCEPTION(ValueError) << "Op " << primitive->name()
                             << " has invalid input. kernel_size expected to be greater than 0 but got "
                             << kernel_size_val << " instead.";
  }
  if (stride_val < 1) {
    MS_EXCEPTION(ValueError) << "Op " << primitive->name()
                             << " has invalid input. stride expected to be greater than 0 but got " << stride_val
                             << " instead.";
  }

  const int64_t kNum2 = 2;
  if (MS_UNLIKELY(kernel_size_val / kNum2 < padding_val)) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name()
                             << ", pad should be at most half of kernel size, but got pad = " << padding_val
                             << " and kernel_size = " << kernel_size_val;
  }

  auto ceil_mode = ceil_mode_opt.value();
  auto output_size = AvgPool1DOutputShapePadLR(last_dim_value, kernel_size_val, stride_val, padding_val, ceil_mode);
  if (output_size <= 0) {
    MS_EXCEPTION(ValueError) << "For Op " << primitive->name() << ", output size should be > 0, but got " << output_size
                             << ".";
  }
  input_shape[input_rank - 1] = output_size;
  return {input_shape};
}

std::vector<TypeId> AvgPool1DFuncImpl::InferType(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  auto input_type = input_infos[kInputIndex0]->GetType();
  return {input_type};
}
}  // namespace ops
}  // namespace mindspore
