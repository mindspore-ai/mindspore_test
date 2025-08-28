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
#include "view/diagonal_strides_calc.h"
#include <memory>
#include <algorithm>
#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"

namespace mindspore::ops {
namespace {
inline static int64_t ComputeData(int64_t offset, int64_t dim1, int64_t dim2, std::vector<int64_t> old_shape) {
  int64_t diag_size;
  if (offset >= 0) {
    diag_size = std::max<int64_t>(std::min(old_shape[dim1], old_shape[dim2] - offset), 0);
  } else {
    diag_size = std::max<int64_t>(std::min(old_shape[dim1] + offset, old_shape[dim2]), 0);
  }
  return diag_size;
}
}  // namespace

TensorStorageInfoPtrList DiagonalStridesCalc(const OldTensorInfoPtr old_tensor_info, const int64_t &offset,
                                             const int64_t &dim_1, const int64_t &dim_2) {
  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  auto storage_offset = old_tensor_info->old_offset;
  int64_t dim_size = SizeToLong(old_shape.size());
  auto dim1 = DynamicDimWrap(dim_1, dim_size);
  auto dim2 = DynamicDimWrap(dim_2, dim_size);
  if (dim1 == dim2) {
    MS_EXCEPTION(ValueError) << "For 'Diagonal', dim1 and dim2 cannot be identical, but got : dim1 =" << dim1
                             << " and dim2 = " << dim2 << ".";
  }
  constexpr int64_t kDimNum = 2;
  if (dim_size < kDimNum) {
    MS_EXCEPTION(ValueError) << "For 'Diagonal', input must be at least 2-dimensional, but got : " << dim_size << ".";
  }

  auto new_shape = old_shape;
  auto new_strides = old_strides;
  int64_t diag_size = ComputeData(offset, dim1, dim2, old_shape);
  if (diag_size == 0) {
    // skip
  } else if (offset >= 0) {
    storage_offset += LongToSize(offset * old_strides[dim2]);
  } else {
    storage_offset += LongToSize(-offset * old_strides[dim1]);
  }

  new_shape.erase(new_shape.begin() + std::max(dim1, dim2));
  new_strides.erase(new_strides.begin() + std::max(dim1, dim2));
  new_shape.erase(new_shape.begin() + std::min(dim1, dim2));
  new_strides.erase(new_strides.begin() + std::min(dim1, dim2));
  new_shape.push_back(diag_size);
  new_strides.push_back(old_strides[dim1] + old_strides[dim2]);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}

TensorStorageInfoPtrList DiagonalBasicTypeCalc(const mindspore::tensor::TensorPtr &input_tensor, const int64_t &offset,
                                               const int64_t &dim1, const int64_t &dim2) {
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  return DiagonalStridesCalc(old_tensor_info, offset, dim1, dim2);
}

TensorStorageInfoPtrList DiagonalCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto offset = GetValue<int64_t>(inputs[kInputIndex1]);
  auto dim1 = GetValue<int64_t>(inputs[kInputIndex2]);
  auto dim2 = GetValue<int64_t>(inputs[kInputIndex3]);
  return DiagonalBasicTypeCalc(input_tensor, offset, dim1, dim2);
}

REG_VIEW_STRIDES_CALC_FUN(Diagonal, DiagonalCalc);
}  // namespace mindspore::ops
