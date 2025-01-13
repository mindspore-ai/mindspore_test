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

TensorStorageInfoPtrList TransposeExtStridesCalc(const OldTensorInfoPtr old_tensor_info, const int64_t &dim0,
                                                 const int64_t &dim1) {
  auto oldShape = old_tensor_info->old_shape;
  auto oldStrides = old_tensor_info->old_strides;
  auto oldStorageOffset = old_tensor_info->old_offset;
  int64_t dim_size = SizeToLong(oldShape.size());
  // if x is a scalar tensor, then dim must be in the range [-1, 0].
  if (dim_size <= 0) {
    dim_size = 1;
  }
  if (dim0 < -dim_size || dim0 >= dim_size || dim1 < -dim_size || dim1 >= dim_size) {
    MS_EXCEPTION(ValueError) << "For primitive[TransposeExt], the dim1 must be in [" << -dim_size << ", " << dim_size
                             << "], but got dim0 " << dim0 << ", dim1 " << dim1;
  }
  auto dim0_new = DynamicDimWrap(dim0, dim_size);
  auto dim1_new = DynamicDimWrap(dim1, dim_size);
  if (dim0_new == dim1_new) {
    bool is_contiguous = IsContiguous(oldShape, oldStrides);
    auto newStorageInfo = std::make_shared<TensorStorageInfo>(
      oldShape, oldStrides, oldStorageOffset, old_tensor_info->ori_shape, old_tensor_info->ori_strides, is_contiguous);
    return {newStorageInfo};
  }

  ShapeVector newShape = oldShape;
  StridesVecotr newStrides = oldStrides;
  std::swap(newShape[dim0_new], newShape[dim1_new]);
  std::swap(newStrides[dim0_new], newStrides[dim1_new]);
  bool is_contiguous = IsContiguous(newShape, newStrides);
  auto newStorageInfo = std::make_shared<TensorStorageInfo>(
    newShape, newStrides, oldStorageOffset, old_tensor_info->ori_shape, old_tensor_info->ori_strides, is_contiguous);
  return {newStorageInfo};
}

TensorStorageInfoPtrList TransposeExtCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (CheckInputsNull(inputs, kTransposeExtCalcInputsNum) || !inputs[kIndex0]->isa<tensor::BaseTensor>() ||
      !inputs[kIndex1]->isa<IntegerImm>() || !inputs[kIndex2]->isa<IntegerImm>()) {
    return {};
  }
  auto tensor = inputs[kIndex0]->cast<tensor::BaseTensorPtr>();
  auto dim0 = GetValue<int64_t>(inputs[kIndex1]);
  auto dim1 = GetValue<int64_t>(inputs[kIndex2]);
  auto old_tensor_info = GetOldTensorInfo(tensor);
  return TransposeExtStridesCalc(old_tensor_info, dim0, dim1);
}

REG_VIEW_STRIDES_CALC_FUN(TransposeExt, TransposeExtCalc);
}  // namespace mindspore::ops
