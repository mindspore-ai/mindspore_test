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
#include <set>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "view/slice_ext_strides_calc.h"

namespace {
constexpr size_t kSliceExtInputsNum = 5;
}

namespace mindspore::ops {

TensorStorageInfoPtrList SliceExtStridesCalc(const OldTensorInfoPtr old_tensor_info, const int64_t ori_dim,
                                             const int64_t ori_start, const int64_t ori_end, const int64_t step) {
  MS_CHECK_VALUE(step > 0, "slice step must be positive");

  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;

  int dim_size = SizeToLong(old_shape.size());
  MS_CHECK_VALUE(dim_size > 0, "slice cannot be applied to a 0-dim tensor.");

  auto dim = DynamicDimWrap(ori_dim, dim_size);
  auto dim_value = old_shape[dim];

  auto start = ori_start < 0 ? ori_start + dim_value : ori_start;

  auto end = ori_end < 0 ? ori_end + dim_value : ori_end;

  if (start < 0) {
    start = 0;
  } else if (start > dim_value) {
    start = dim_value;
  }

  if (end < start) {
    end = start;
  } else if (end > dim_value) {
    end = dim_value;
  }

  auto len = end - start;

  auto new_shape = old_shape;
  new_shape[dim] = (len + step - 1) / step;
  auto new_strides = old_strides;
  new_strides[dim] *= step;
  size_t new_storage_offset = old_tensor_info->old_offset + LongToSize(start * old_strides[dim]);

  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}

TensorStorageInfoPtrList SliceExtCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);

  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(old_tensor_info);

  auto dim = GetValue<int64_t>(inputs[kInputIndex1]);
  auto start = GetValue<int64_t>(inputs[kInputIndex2]);
  auto end = GetValue<int64_t>(inputs[kInputIndex3]);
  auto step = GetValue<int64_t>(inputs[kInputIndex4]);

  return SliceExtStridesCalc(old_tensor_info, dim, start, end, step);
}

REG_VIEW_STRIDES_CALC_FUN(SliceExt, SliceExtCalc);
}  // namespace mindspore::ops
