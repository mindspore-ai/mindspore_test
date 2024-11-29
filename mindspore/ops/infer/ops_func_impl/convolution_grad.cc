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

#include "infer/ops_func_impl/convolution_grad.h"
#include <string>
#include <set>
#include "utils/check_convert_utils.h"
#include "mindspore/ops/op_def/op_name.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kConvolutionGradInputArgsSize = 11;
constexpr size_t kConvolutionGradInputDims = 4;
}  // namespace
ShapeArray ConvolutionGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                               const InferInfoPtrList &input_infos) const {
  auto x_shape = input_infos[kInputIndex1]->GetShape();
  auto weight_shape = input_infos[kInputIndex2]->GetShape();
  auto dout_shape = input_infos[kInputIndex0]->GetShape();

  auto get_bias_grad_shape = [dout_shape]() {
    if (IsDynamicRank(dout_shape) || IsDynamic(dout_shape)) {
      return abstract::Shape::kShapeDimAny;
    }
    return dout_shape[1];
  };

  ShapeVector bias_grad_shape = {get_bias_grad_shape()};
  return {x_shape, weight_shape, bias_grad_shape};
}

std::vector<TypeId> ConvolutionGradFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  auto x_type_ptr = input_infos[kInputIndex1]->GetType();
  auto weight_type_ptr = input_infos[kInputIndex2]->GetType();
  return {x_type_ptr, weight_type_ptr, x_type_ptr};
}
}  // namespace ops
}  // namespace mindspore
