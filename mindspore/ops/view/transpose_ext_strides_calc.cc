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

#include "view/transpose_ext_strides_calc.h"
#include <vector>
#include <memory>
#include <utility>
#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"
#include "mindspore/ops/op_def/op_name.h"

namespace mindspore::ops {
constexpr size_t kTransposeExtCalcInputsNum = 3;

TensorStorageInfoPtrList TransposeExtCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kTransposeExtCalcInputsNum) || !inputs[kIndex0]->isa<tensor::BaseTensor>() ||
      !inputs[kIndex1]->isa<IntegerImm>() || !inputs[kIndex2]->isa<IntegerImm>()) {
    return {};
  }
  auto tensor = inputs[kIndex0]->cast<tensor::BaseTensorPtr>();
  auto dim0 = GetValue<int64_t>(inputs[kIndex1]);
  auto dim1 = GetValue<int64_t>(inputs[kIndex2]);
  auto old_tensor_info = GetOldTensorInfo(tensor);
  auto oldShape = old_tensor_info->old_shape;
  auto oldStrides = old_tensor_info->old_strides;
  auto oldStorageOffset = old_tensor_info->old_offset;
  int64_t dim_size = SizeToLong(oldShape.size());
  // if x is a scalar tensor, then dim must be in the range [-1, 0].
  if (dim_size <= 0) {
    dim_size = 1;
  }
  MS_CHECK_VALUE(dim0 >= -dim_size && dim0 < dim_size,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("dim0", dim0, kIncludeLeft, {-dim_size, dim_size}, prim));
  MS_CHECK_VALUE(dim1 >= -dim_size && dim1 < dim_size,
                 CheckAndConvertUtils::FormatCheckInRangeMsg("dim1", dim1, kIncludeLeft, {-dim_size, dim_size}, prim));
  dim0 = DynamicDimWrap(dim0, dim_size);
  dim1 = DynamicDimWrap(dim1, dim_size);
  if (dim0 == dim1) {
    bool is_contiguous = IsContiguous(oldShape, oldStrides);
    auto newStorageInfo = std::make_shared<TensorStorageInfo>(
      oldShape, oldStrides, oldStorageOffset, old_tensor_info->ori_shape, old_tensor_info->ori_strides, is_contiguous);
    return {newStorageInfo};
  }

  ShapeVector newShape = oldShape;
  StridesVecotr newStrides = oldStrides;
  std::swap(newShape[dim0], newShape[dim1]);
  std::swap(newStrides[dim0], newStrides[dim1]);
  bool is_contiguous = IsContiguous(newShape, newStrides);
  auto newStorageInfo = std::make_shared<TensorStorageInfo>(
    newShape, newStrides, oldStorageOffset, old_tensor_info->ori_shape, old_tensor_info->ori_strides, is_contiguous);
  return {newStorageInfo};
}

REG_VIEW_STRIDES_CALC_FUN(TransposeExt, TransposeExtCalc);
}  // namespace mindspore::ops
