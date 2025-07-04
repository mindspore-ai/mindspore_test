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

#include "view/transpose_strides_calc.h"
#include <vector>
#include <memory>
#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"

namespace mindspore::ops {

TensorStorageInfoPtrList TransposeStridesCalc(const OldTensorInfoPtr old_tensor_info,
                                              const std::vector<int64_t> &input_perm) {
  const auto &old_shape = old_tensor_info->old_shape;
  const auto &old_strides = old_tensor_info->old_strides;
  const auto &old_storage_offset = old_tensor_info->old_offset;

  auto &dims = input_perm;
  const auto ndim = old_shape.size();

  ShapeVector new_shape(ndim);
  std::vector<int64_t> new_strides(ndim);
  std::vector<bool> seen_dims(ndim, false);

  for (size_t i = 0; i < ndim; i++) {
    const auto wrap_dim = DynamicDimWrap(dims[i], ndim);
    if (seen_dims[wrap_dim]) {
      MS_EXCEPTION(ValueError) << CheckAndConvertUtils::FormatCommMsg(
        "For 'Transpose' perms should all be unique dim, but", wrap_dim, " is not unique!");
    }
    seen_dims[wrap_dim] = true;
    new_shape[i] = old_shape[wrap_dim];
    new_strides[i] = old_strides[wrap_dim];
  }

  bool is_contiguous = IsContiguous(new_shape, new_strides);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, old_storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, is_contiguous);
  return {new_storage_info};
}

TensorStorageInfoPtrList TransposeBasicTypeCalc(const PrimitivePtr &prim,
                                                const mindspore::tensor::TensorPtr &input_tensor,
                                                const std::vector<int64_t> &dims) {
  MS_EXCEPTION_IF_NULL(input_tensor);
  const auto &x_shape = input_tensor->shape();
  auto x_rank = x_shape.size();
  MS_CHECK_VALUE(dims.size() == x_rank, CheckAndConvertUtils::FormatCommMsg(
                                          "For '", prim->name(), "', size of perm should equal to rank of x, but got ",
                                          dims.size(), " and ", x_rank, "!"));
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  return TransposeStridesCalc(old_tensor_info, dims);
}

TensorStorageInfoPtrList TransposeCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (!inputs[kInputIndex0]->isa<tensor::Tensor>() || !inputs[kInputIndex1]->isa<ValueSequence>()) {
    return {};
  }
  auto tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  const auto &dims = GetValue<std::vector<int64_t>>(inputs[1]);
  return TransposeBasicTypeCalc(prim, tensor, dims);
}

REG_VIEW_STRIDES_CALC_FUN(Transpose, TransposeCalc);
}  // namespace mindspore::ops
