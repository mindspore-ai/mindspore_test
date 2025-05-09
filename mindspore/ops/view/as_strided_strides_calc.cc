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

#include "view/as_strided_strides_calc.h"
#include <vector>
#include <memory>

namespace mindspore::ops {
constexpr size_t kAsStridedInputsNum = 4;
TensorStorageInfoPtrList AsStridedCalcImpl(const PrimitivePtr &prim, const tensor::TensorPtr &input,
                                           const std::vector<int64_t> &size, const std::vector<int64_t> &stride,
                                           int64_t offset) {
  MS_EXCEPTION_IF_NULL(input);
  auto old_tensor_info = GetOldTensorInfo(input);
  // To do check
  auto new_storage_info = std::make_shared<TensorStorageInfo>(size, stride, offset, old_tensor_info->ori_shape,
                                                              old_tensor_info->ori_strides, IsContiguous(size, stride));
  return {new_storage_info};
}

TensorStorageInfoPtrList AsStridedCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (inputs.size() != kAsStridedInputsNum) {
    return {};
  }
  auto input_tensor = inputs[0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto shape = GetValue<std::vector<int64_t>>(inputs[1]);
  if (std::any_of(shape.begin(), shape.end(), [](const int &shape_i) { return shape_i < -1; })) {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim->name()
                             << "], the component of shape can't be less than -1, but got " << shape;
  }
  auto stride = GetValue<std::vector<int64_t>>(inputs[2]);
  auto storage_offset = GetValue<int64_t>(inputs[3]);
  return AsStridedCalcImpl(prim, input_tensor, shape, stride, storage_offset);
}
}  // namespace mindspore::ops
