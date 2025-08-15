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

#include "view/view_strides_calc.h"
#include <vector>
#include <memory>
#include <utility>
#include "view/reshape_strides_calc.h"

namespace mindspore::ops {
TensorStorageInfoPtrList ViewStridesCalc(const std::vector<int64_t> &cur_shape, const std::vector<int64_t> &cur_strides,
                                         const TensorStorageInfoPtr &cur_storage_info,
                                         const std::vector<int64_t> &shape) {
  TensorStorageInfoPtrList storage_info_list;
  auto new_storage_info = ReshapeStridesCalc(cur_shape, cur_strides, cur_storage_info, shape);
  if (MS_LIKELY(new_storage_info)) {
    storage_info_list.push_back(std::move(new_storage_info));
  }
  return storage_info_list;
}

TensorStorageInfoPtrList ViewBasicTypeCalc(const tensor::TensorPtr &input_tensor, const std::vector<int64_t> &shape) {
  return ViewStridesCalc(input_tensor->shape(), input_tensor->stride(), input_tensor->storage_info(), shape);
}

TensorStorageInfoPtrList ViewCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto shape = GetValue<std::vector<int64_t>>(inputs[kInputIndex1]);
  return ViewBasicTypeCalc(input_tensor, shape);
}
}  // namespace mindspore::ops
