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

#include "infer/ops_func_impl/adaptive_max_pool2d.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
const int64_t kUserInputChangeDim = -2;
int64_t GetDimOfOutputSizeDim(const InferInfoPtr &output_size, const int index) {
  auto &output_size_info_ptr = output_size;
  auto output_size_opt = output_size_info_ptr->GetArrayValue<int64_t>();
  if (!output_size_opt.has_value()) {
    return abstract::Shape::kShapeDimAny;
  }

  auto output_size_array_value = output_size_opt.value();
  if (output_size_array_value.IsValueUnknown(index)) {
    return abstract::Shape::kShapeDimAny;
  }

  int64_t val = static_cast<int64_t>(output_size_array_value[index]);
  if (val < -1) {
    MS_LOG(EXCEPTION) << "The output_size input is invalid";
  }

  if (val == -1) {
    return kUserInputChangeDim;
  }
  return val;
}

int64_t GetDimOfInputDim(const InferInfoPtr &tensor, const int index) {
  ShapeVector tensor_shape = tensor->GetShape();
  if (tensor_shape[index] != abstract::Shape::kShapeDimAny) {
    return tensor_shape[index];
  }
  return abstract::Shape::kShapeDimAny;
}
}  // namespace
ShapeArray AdaptiveMaxPool2DFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const InferInfoPtrList &input_infos) const {
  auto &x_tensor = input_infos[kInputIndex0];
  auto x_shape = x_tensor->GetShape();

  if (x_tensor->IsDynamicRank()) {
    ShapeVector shape1 = {abstract::Shape::kShapeRankAny};
    ShapeVector shape2 = {abstract::Shape::kShapeRankAny};
    return {shape1, shape2};
  }

  int64_t shapeDimNums = SizeToLong(x_shape.size());
  const int64_t kInputNumDims3 = 3;
  const int64_t kInputNumDims4 = 4;
  auto prim_name = primitive->name();
  CheckAndConvertUtils::CheckInRange("rank of x", shapeDimNums, kIncludeBoth, {kInputNumDims3, kInputNumDims4},
                                     prim_name);
  int64_t output_size_dim0 = GetDimOfOutputSizeDim(input_infos[kInputIndex1], kInputIndex0);
  int64_t output_size_dim1 = GetDimOfOutputSizeDim(input_infos[kInputIndex1], kInputIndex1);
  if (kInputNumDims3 == shapeDimNums) {
    int64_t input_dim0 = GetDimOfInputDim(input_infos[kInputIndex0], kInputIndex0);
    int64_t input_dim1 = GetDimOfInputDim(input_infos[kInputIndex0], kInputIndex1);
    int64_t input_dim2 = GetDimOfInputDim(input_infos[kInputIndex0], kInputIndex2);
    if (output_size_dim0 == kUserInputChangeDim) {
      output_size_dim0 = input_dim1;
    }
    if (output_size_dim1 == kUserInputChangeDim) {
      output_size_dim1 = input_dim2;
    }
    ShapeVector shape1 = {input_dim0, output_size_dim0, output_size_dim1};
    ShapeVector shape2 = {input_dim0, output_size_dim0, output_size_dim1};
    return {shape1, shape2};
  }
  int64_t input_dim0 = GetDimOfInputDim(input_infos[kInputIndex0], kInputIndex0);
  int64_t input_dim1 = GetDimOfInputDim(input_infos[kInputIndex0], kInputIndex1);
  int64_t input_dim2 = GetDimOfInputDim(input_infos[kInputIndex0], kInputIndex2);
  int64_t input_dim3 = GetDimOfInputDim(input_infos[kInputIndex0], kInputIndex3);
  if (output_size_dim0 == kUserInputChangeDim) {
    output_size_dim0 = input_dim2;
  }
  if (output_size_dim1 == kUserInputChangeDim) {
    output_size_dim1 = input_dim3;
  }
  ShapeVector shape1 = {input_dim0, input_dim1, output_size_dim0, output_size_dim1};
  ShapeVector shape2 = {input_dim0, input_dim1, output_size_dim0, output_size_dim1};
  return {shape1, shape2};
}

std::vector<TypeId> AdaptiveMaxPool2DFuncImpl::InferType(const PrimitivePtr &primitive,
                                                         const InferInfoPtrList &input_infos) const {
  return {input_infos[kInputIndex0]->GetType(), kNumberTypeInt64};
}
}  // namespace ops
}  // namespace mindspore
