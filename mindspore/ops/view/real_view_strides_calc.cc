/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "view/real_view_strides_calc.h"

namespace mindspore::ops {

TensorStorageInfoPtrList RealImagViewBasicTypeCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs,
                                                   bool is_real, bool is_complex_data_type) {
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  MS_EXCEPTION_IF_NULL(old_tensor_info);

  auto old_shape = old_tensor_info->old_shape;
  auto old_strides = old_tensor_info->old_strides;
  size_t old_storage_offset = old_tensor_info->old_offset;
  int dim_size = SizeToLong(old_shape.size());

  auto new_shape = old_shape;
  if (is_complex_data_type && dim_size > 0) {
    // we reexplained the complex storage into two parts, so the old shape is doubled, eg. complex64 to 2 float32
    old_shape[dim_size - 1] *= 2;
    old_strides[dim_size - 1] = 1;
    for (int i = dim_size - 2; i >= 0; i--) {
      old_strides[i] = old_strides[i + 1] * old_shape[i + 1];
    }
  }

  auto new_strides = old_strides;
  auto new_storage_offset = old_storage_offset;

  if (is_complex_data_type) {
    // if dim_size is 0, the complex tensor is a scalar, so we don't need to calculate the strides
    if (dim_size > 0) {
      // complex has two parts, real and imag, they are stored one after another, so the stride is 2
      new_strides[dim_size - 1] = 2;
    }
    // real is first, imag is second
    new_storage_offset = is_real ? old_storage_offset : old_storage_offset + 1;
  }

  return {std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_storage_offset, old_shape, old_strides,
                                              IsContiguous(old_shape, new_strides))};
}

TensorStorageInfoPtrList RealViewCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  return RealImagViewBasicTypeCalc(prim, inputs, true, true);
}

REG_VIEW_STRIDES_CALC_FUN(RealView, RealViewCalc);

}  // namespace mindspore::ops
