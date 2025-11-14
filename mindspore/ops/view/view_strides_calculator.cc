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
#include "view/view_strides_calculator.h"
#include <vector>
#include <memory>
#include <functional>
#include <numeric>

namespace mindspore::ops {
ViewStridesCalcFactory &ViewStridesCalcFactory::GetInstance() {
  static ViewStridesCalcFactory instance;
  return instance;
}

bool IsDynamic(const std::vector<int64_t> &shape) {
  return std::any_of(shape.begin(), shape.end(), [](int64_t value) { return value < 0; });
}

bool HasZero(const std::vector<int64_t> &value) {
  for (size_t i = 0; i < value.size(); ++i) {
    if (value[i] == 0) {
      return true;
    }
  }
  return false;
}

bool CheckInputsNull(const std::vector<ValuePtr> &inputs, const size_t &input_num) {
  if (inputs.size() != input_num) {
    MS_LOG(DEBUG) << "inputs.size() is not equal to input_num, inputs.size():" << inputs.size()
                  << " input_num:" << input_num;
    return true;
  }

  return std::any_of(inputs.cbegin(), inputs.cend(), [](const ValuePtr &v) { return v == nullptr; });
}

std::vector<int64_t> GetOriStrides(const std::vector<int64_t> &shape) {
  if (shape.empty()) {
    return {};
  }

  std::vector<int64_t> ret(shape.size(), 1);
  int64_t strides = 1;
  for (size_t i = shape.size() - 1; i > 0; --i) {
    strides *= shape[i];
    ret[i - 1] = strides;
  }
  return ret;
}

bool IsContiguous(const ShapeVector &shape, const std::vector<int64_t> &strides) {
  if (shape.size() != strides.size()) {
    MS_LOG(EXCEPTION) << "shape.size() != strides.size()";
  }

  auto numel = std::accumulate(shape.begin(), shape.end(), static_cast<int64_t>(1), std::multiplies<int64_t>());
  if (numel == 0) {
    return true;
  }

  int64_t z = 1;
  for (int64_t i = SizeToLong(shape.size()) - 1; i >= 0; --i) {
    const auto &shape_i = shape[i];
    if (shape_i != 1) {
      if (strides[i] == z) {
        z *= shape_i;
      } else {
        return false;
      }
    }
  }

  return true;
}

int64_t DynamicDimWrap(int64_t dim, int64_t dim_post_expr, bool wrap_scalar) {
  if (MS_LIKELY(dim_post_expr * -1 <= dim && dim < dim_post_expr)) {
    if (dim < 0) {
      return dim + dim_post_expr;
    }
    return dim;
  }
  if (dim_post_expr == 0) {
    if ((!wrap_scalar)) {
      MS_EXCEPTION(ValueError) << "dim value specified as " << dim << ", but tensor has no dimensions";
    }
    return DynamicDimWrap(dim, 1, false);
  }
  MS_EXCEPTION(ValueError) << "Dimension out of range (expected to be in range of [" << -dim_post_expr << ", "
                           << dim_post_expr << "), but got " << dim << ")";
}

OldTensorInfoPtr GetOldTensorInfo(const tensor::TensorPtr &tensor) {
  if (tensor->storage_info() == nullptr) {
    auto old_strides = GetOriStrides(tensor->shape());
    return std::make_shared<OldTensorInfo>(tensor->shape(), old_strides, tensor->shape(), old_strides, 0);
  } else {
    auto storage_info = tensor->storage_info();
    return std::make_shared<OldTensorInfo>(storage_info->shape, storage_info->strides, storage_info->ori_shape,
                                           storage_info->ori_strides, storage_info->storage_offset);
  }
}
}  // namespace mindspore::ops
