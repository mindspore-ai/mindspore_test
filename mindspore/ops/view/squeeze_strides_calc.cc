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
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
namespace {
constexpr size_t kSqueezeCalcInputsNum = 2;
constexpr auto kSqueezedNum = 1;
}  // namespace
TensorStorageInfoPtrList SqueezeCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kSqueezeCalcInputsNum) || !inputs[kInputIndex0]->isa<tensor::BaseTensor>() ||
      (!inputs[kInputIndex1]->isa<ValueSequence>() && !inputs[kInputIndex1]->isa<None>())) {
    return {};
  }
  auto tensor = inputs[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(tensor);
  auto old_tensor_info = GetOldTensorInfo(tensor);
  auto oldShape = old_tensor_info->old_shape;
  auto oldStrides = old_tensor_info->old_strides;
  auto oldStorageOffset = old_tensor_info->old_offset;
  const auto ndims = oldShape.size();

  if (ndims == 0) {
    bool is_contiguous = IsContiguous(oldShape, oldStrides);
    auto newStorageInfo = std::make_shared<TensorStorageInfo>(
      oldShape, oldStrides, oldStorageOffset, old_tensor_info->ori_shape, old_tensor_info->ori_strides, is_contiguous);
    return {newStorageInfo};
  }

  std::vector<bool> seen_dims(ndims, false);
  if (inputs[kInputIndex1]->isa<None>()) {
    for (size_t i = 0; i < ndims; i++) {
      seen_dims[i] = true;
    }
  } else {
    const auto &axis = GetValue<std::vector<int64_t>>(inputs[kInputIndex1]);
    if (!axis.empty() || axis.size() == 0) {
      for (size_t i = 0; i < axis.size(); i++) {
        CheckAndConvertUtils::CheckInRange<int64_t>("element or value of axis", axis[i], kIncludeLeft, {-ndims, ndims},
                                                    "Squeeze");
        const auto wrap_dim = DynamicDimWrap(axis[i], ndims);
        MS_EXCEPTION_IF_CHECK_FAIL(
          !seen_dims[wrap_dim],
          "For Squeeze, wrap_dim appears multiple times in the list of dim, which is not allowed.");
        seen_dims[wrap_dim] = true;
      }
    }
  }
  // delete shape dim if it equals one in seen dimension.
  ShapeVector newShape;
  StridesVecotr newStrides;
  for (size_t i = 0; i < ndims; i++) {
    if (!seen_dims[i] || oldShape[i] != kSqueezedNum) {
      newShape.push_back(oldShape[i]);
      newStrides.push_back(oldStrides[i]);
    }
  }

  bool is_contiguous = IsContiguous(newShape, newStrides);
  auto newStorageInfo = std::make_shared<TensorStorageInfo>(
    newShape, newStrides, oldStorageOffset, old_tensor_info->ori_shape, old_tensor_info->ori_strides, is_contiguous);
  return {newStorageInfo};
}
REG_VIEW_STRIDES_CALC_FUN(Squeeze, SqueezeCalc);
}  // namespace mindspore::ops
