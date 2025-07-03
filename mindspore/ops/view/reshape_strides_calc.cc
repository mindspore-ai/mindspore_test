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

#include "view/reshape_strides_calc.h"
#include <algorithm>
#include <vector>
#include <memory>
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
ShapeVector ReshapeUpdateShape(const ShapeVector &input_shape, ShapeVector shape) {
  int64_t x_num = 1;
  for (int64_t value : input_shape) {
    x_num = LongMulWithOverflowCheck(value, x_num);
  }

  auto it_first = find(shape.begin(), shape.end(), -1);
  if (it_first != shape.end()) {
    auto it_second = find(it_first + 1, shape.end(), -1);
    if (it_second != shape.end()) {
      MS_EXCEPTION(ValueError) << "At most one component of input shape can be -1, but got " << shape;
    }
    auto index = LongToSize(std::distance(shape.begin(), it_first));
    int64_t infer_value = x_num;
    for (size_t i = 0; i < shape.size(); ++i) {
      int64_t value = shape[i];
      if (value != -1 && value != 0) {
        infer_value = infer_value / value;
      }
    }
    shape[index] = infer_value;
  }

  int64_t shape_num = 1;
  for (int64_t value : shape) {
    shape_num = LongMulWithOverflowCheck(value, shape_num);
  }
  if (shape_num != x_num) {
    MS_EXCEPTION(ValueError) << "The accumulate of x_shape must be equal to out_shape, but got x_shape: " << input_shape
                             << ", and out_shape: " << shape;
  }
  return shape;
}

std::optional<std::vector<int64_t>> ComputeUncontiguousStrides(const std::vector<int64_t> &old_shape,
                                                               const std::vector<int64_t> &old_strides,
                                                               const std::vector<int64_t> &new_shape) {
  if (old_shape.empty()) {
    return std::vector<int64_t>(new_shape.size(), 1);
  }

  bool is_old_empty = std::any_of(old_shape.begin(), old_shape.end(), [](const int64_t dim) { return dim == 0; });
  if (is_old_empty && old_shape == new_shape) {
    return old_strides;
  }

  int64_t new_rank = SizeToLong(new_shape.size());
  std::vector<int64_t> new_strides(new_rank, 0);
  if (is_old_empty) {
    for (int64_t dim = new_rank - 1; dim >= 0; --dim) {
      if (dim == (new_rank - 1)) {
        new_strides[dim] = 1;
      } else {
        new_strides[dim] = std::max(new_shape[dim + 1], static_cast<int64_t>(1)) * new_strides[dim + 1];
      }
    }
    return new_strides;
  }

  int64_t view_dim = new_rank - 1;
  int64_t base_stride = old_strides.back();
  int64_t tensor_elems = 1;
  int64_t view_elems = 1;

  for (int64_t dim = SizeToLong(old_shape.size()) - 1; dim >= 0; --dim) {
    tensor_elems *= old_shape[dim];
    if ((dim == 0) || (old_shape[dim - 1] != 1 && old_strides[dim - 1] != tensor_elems * base_stride)) {
      while (view_dim >= 0 && (view_elems < tensor_elems || new_shape[view_dim] == 1)) {
        new_strides[view_dim] = view_elems * base_stride;
        view_elems *= new_shape[view_dim];
        --view_dim;
      }
      if (view_elems != tensor_elems) {
        return std::nullopt;
      }
      if (dim > 0) {
        base_stride = old_strides[dim - 1];
        tensor_elems = 1;
        view_elems = 1;
      }
    }
  }
  if (view_dim != -1) {
    return std::nullopt;
  }

  return new_strides;
}

TensorStorageInfoPtrList ReshapeUncontiguousCalcImpl(const mindspore::ops::OldTensorInfoPtr &old_tensor_info,
                                                     const std::vector<int64_t> &shape) {
  const auto &old_shape = old_tensor_info->old_shape;
  const auto &old_strides = old_tensor_info->old_strides;
  auto storage_offset = old_tensor_info->old_offset;
  const auto &new_shape = ReshapeUpdateShape(old_shape, shape);
  const auto &new_strides_opt = ComputeUncontiguousStrides(old_shape, old_strides, new_shape);
  if (new_strides_opt.has_value()) {
    const auto &new_strides = new_strides_opt.value();
    auto new_storage_info = std::make_shared<TensorStorageInfo>(
      new_shape, new_strides, storage_offset, old_tensor_info->ori_shape, old_tensor_info->ori_strides, false);
    return {new_storage_info};
  }

  return {};
}

TensorStorageInfoPtrList ReshapeCalcImpl(const mindspore::ops::OldTensorInfoPtr &old_tensor_info,
                                         const std::vector<int64_t> &shape) {
  const auto &old_shape = old_tensor_info->old_shape;
  auto storage_offset = old_tensor_info->old_offset;
  const auto &new_shape = ReshapeUpdateShape(old_shape, shape);
  const auto &new_strides = GetOriStrides(new_shape);
  auto new_storage_info =
    std::make_shared<TensorStorageInfo>(new_shape, new_strides, storage_offset, old_tensor_info->ori_shape,
                                        old_tensor_info->ori_strides, IsContiguous(new_shape, new_strides));
  return {new_storage_info};
}

TensorStorageInfoPtrList ReshapeCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  auto input_tensor = inputs[kInputIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(input_tensor);
  auto input_type = input_tensor->Dtype();
  (void)CheckAndConvertUtils::CheckTypeValid("input", input_type, common_valid_types_with_complex_and_bool,
                                             prim->name());

  auto shape = GetValue<std::vector<int64_t>>(inputs[kInputIndex1]);
  if (std::any_of(shape.begin(), shape.end(), [](const int &shape_i) { return shape_i < -1; })) {
    MS_EXCEPTION(ValueError) << "For primitive[" << prim->name()
                             << "], the component of shape can't be less than -1, but got " << shape;
  }

  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  auto ori_storage_info = input_tensor->storage_info();
  if (ori_storage_info != nullptr && !ori_storage_info->is_contiguous) {
    return ReshapeUncontiguousCalcImpl(old_tensor_info, shape);
  }

  return ReshapeCalcImpl(old_tensor_info, shape);
}

TensorStorageInfoPtrList ReshapeBasicTypeCalc(const tensor::TensorPtr &input_tensor,
                                              const std::vector<int64_t> &shape) {
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  return ReshapeUncontiguousCalcImpl(old_tensor_info, shape);
}

REG_VIEW_STRIDES_CALC_FUN(Reshape, ReshapeCalc);
}  // namespace mindspore::ops
