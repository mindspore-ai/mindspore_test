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
#include "view/squeeze_strides_calc.h"
#include <vector>
#include <memory>
#include <string>
#include <utility>
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
namespace {
constexpr size_t kSqueezeCalcInputsNum = 2;
constexpr auto kSqueezedNum = 1;
}  // namespace
TensorStorageInfoPtrList SqueezeBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor,
                                              const std::vector<int64_t> &axis) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &old_shape = input_tensor->shape();
  const auto &old_strides = input_tensor->stride();
  auto [ori_shape, ori_strides, storage_offset] =
    GetOriShapeStridesAndOffset(old_shape, old_strides, input_tensor->storage_info());

  const auto ndims = old_shape.size();
  TensorStorageInfoPtrList newStorageInfoList{};
  if (ndims == 0) {
    bool is_contiguous = IsContiguous(old_shape, old_strides);
    newStorageInfoList.push_back(std::make_shared<TensorStorageInfo>(
      old_shape, old_strides, storage_offset, std::move(ori_shape), std::move(ori_strides), is_contiguous));
    return newStorageInfoList;
  }

  std::vector<bool> seen_dims(ndims, false);
  if (axis.empty()) {
    for (size_t i = 0; i < ndims; i++) {
      seen_dims[i] = true;
    }
  } else {
    for (int64_t dim : axis) {
      const auto wrap_dim = DynamicDimWrap(dim, ndims);
      seen_dims[wrap_dim] = true;
    }
  }

  // delete shape dim if it equals one in seen dimension.
  ShapeVector new_shape;
  StridesVecotr new_strides;
  for (size_t i = 0; i < ndims; i++) {
    if (!seen_dims[i] || old_shape[i] != kSqueezedNum) {
      new_shape.push_back(old_shape[i]);
      new_strides.push_back(old_strides[i]);
    }
  }

  bool is_contiguous = IsContiguous(new_shape, new_strides);
  newStorageInfoList.push_back(std::make_shared<TensorStorageInfo>(std::move(new_shape), std::move(new_strides),
                                                                   storage_offset, std::move(ori_shape),
                                                                   std::move(ori_strides), is_contiguous));

  return newStorageInfoList;
}
}  // namespace mindspore::ops
