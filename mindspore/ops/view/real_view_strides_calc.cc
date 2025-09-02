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

  auto new_shape = old_tensor_info->old_shape;
  auto new_strides = old_tensor_info->old_strides;
  auto ori_shape = old_tensor_info->ori_shape;
  auto ori_strides = old_tensor_info->ori_strides;
  size_t old_storage_offset = old_tensor_info->old_offset;
  int dim_size = SizeToLong(new_shape.size());
  int ori_dim_size = SizeToLong(ori_shape.size());

  auto new_storage_offset = old_storage_offset;
  if (is_complex_data_type && dim_size > 0 && ori_dim_size > 0) {
    // if old tensor has shape, and because of the complex storage, the new stride is doubled
    for (int i = 0; i < dim_size; i++) {
      new_strides[i] *= 2;
    }
    // we reexplained the complex storage into two parts, so the original shape is doubled, eg. complex64 to 2 float32
    ori_shape[ori_dim_size - 1] *= 2;
    ori_strides[ori_dim_size - 1] = 1;
    // recalculate the original strides with the new shape
    for (int i = ori_dim_size - 2; i >= 0; i--) {
      ori_strides[i] = ori_strides[i + 1] * ori_shape[i + 1];
    }
    new_storage_offset *= 2;
  }

  // for imag part, the storage offset need to add 1
  if (is_complex_data_type && (!is_real)) {
    new_storage_offset += 1;
  }

  return {
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_storage_offset, ori_shape, ori_strides, false)};
}

TensorStorageInfoPtrList RealViewCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  return RealImagViewBasicTypeCalc(prim, inputs, true, true);
}

REG_VIEW_STRIDES_CALC_FUN(RealView, RealViewCalc);

}  // namespace mindspore::ops
