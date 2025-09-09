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

#include "view/view_dtype_strides_calc.h"
#include <vector>
#include <memory>

namespace mindspore::ops {
constexpr size_t kFloat32Size = 4;
constexpr size_t kFloat64Size = 8;
constexpr size_t kComplex64Size = 8;
constexpr size_t kComplex128Size = 16;
static const std::map<TypeId, size_t> kDtypeToSizeMap = {
  {TypeId::kNumberTypeBool, sizeof(bool)},        {TypeId::kNumberTypeInt8, sizeof(int8_t)},
  {TypeId::kNumberTypeInt16, sizeof(int16_t)},    {TypeId::kNumberTypeInt32, sizeof(int32_t)},
  {TypeId::kNumberTypeInt64, sizeof(int64_t)},    {TypeId::kNumberTypeUInt8, sizeof(uint8_t)},
  {TypeId::kNumberTypeUInt16, sizeof(uint16_t)},  {TypeId::kNumberTypeUInt32, sizeof(uint32_t)},
  {TypeId::kNumberTypeUInt64, sizeof(uint64_t)},  {TypeId::kNumberTypeFloat16, sizeof(float16)},
  {TypeId::kNumberTypeFloat32, kFloat32Size},     {TypeId::kNumberTypeFloat64, kFloat64Size},
  {TypeId::kNumberTypeInt, sizeof(int)},          {TypeId::kNumberTypeBFloat16, sizeof(bfloat16)},
  {TypeId::kNumberTypeComplex64, kComplex64Size}, {TypeId::kNumberTypeComplex128, kComplex128Size}};

size_t GetDtypeSize(TypeId type) {
  auto iter = kDtypeToSizeMap.find(type);
  if (iter == kDtypeToSizeMap.end()) {
    MS_LOG(EXCEPTION) << "Unsupported data type " << type;
  }
  return iter->second;
}

std::vector<int64_t> DownsizingElementSize(const std::vector<int64_t> &old_strides, int64_t size_ratio,
                                           TypeId old_dtype, TypeId new_dtype) {
  const int64_t dim = old_strides.size();
  if (old_strides[dim - 1] != 1) {
      MS_EXCEPTION(ValueError) << "old_strides[-1] must be 1 to view " << TypeIdToString(old_dtype)
                               << " as " << TypeIdToString(new_dtype)
                               << " (different element sizes), but got " << old_strides[dim - 1];
    }
    std::vector<int64_t> new_strides(dim);
    for (int64_t i = 0; i < dim - 1; ++i) {
      new_strides[i] = old_strides[i] * size_ratio;
    }
    new_strides[dim - 1] = 1;
    return new_strides;
}

std::vector<int64_t> UpsizingElementSize(const std::vector<int64_t> &old_strides, int64_t size_ratio,
                                           TypeId old_dtype, TypeId new_dtype) {
  const int64_t dim = old_strides.size();
  if (old_strides[dim - 1] != 1) {
      MS_EXCEPTION(ValueError) << "old_strides[-1] must be 1 to view " << TypeIdToString(old_dtype)
                               << " as " << TypeIdToString(new_dtype)
                               << " (different element sizes), but got " << old_strides[dim - 1];
    }
    std::vector<int64_t> new_strides(dim);
    for (int64_t i = 0; i < dim - 1; ++i) {
      if (old_strides[i] % size_ratio != 0) {
        MS_EXCEPTION(ValueError) << "old_strides[" << i << "] must be divisible by " << size_ratio
                                 << " to view " << TypeIdToString(old_dtype) << " as " << TypeIdToString(new_dtype)
                                 << " (different element sizes), but got " << old_strides[i];
      }
      new_strides[i] = old_strides[i] / size_ratio;
    }
    new_strides[dim - 1] = 1;
    return new_strides;
}

BasicCalcResult ViewDtypeBasicTypeCalc(const tensor::TensorPtr &input_tensor, const int64_t &dtype) {
  MS_EXCEPTION_IF_NULL(input_tensor);

  TypeId old_dtype = input_tensor->data_type();
  TypeId new_dtype = static_cast<TypeId>(dtype);

  if (old_dtype == new_dtype) {
    return {input_tensor->storage_info(), old_dtype};
  }

  int64_t old_element_size = GetDtypeSize(old_dtype);
  int64_t new_element_size = GetDtypeSize(new_dtype);
  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  const auto &old_shape = old_tensor_info->old_shape;
  const auto &old_strides = old_tensor_info->old_strides;
  int64_t old_offset = old_tensor_info->old_offset;
  const auto &ori_shape = old_tensor_info->ori_shape;
  const auto &ori_strides = old_tensor_info->ori_strides;

  if (old_element_size == new_element_size) {
    return {std::make_shared<TensorStorageInfo>(old_shape, old_strides, old_offset, ori_shape, ori_strides,
                                                IsContiguous(old_shape, old_strides)), new_dtype};
  }

  if (old_shape.empty()) {
    MS_EXCEPTION(ValueError) << "Input dim cannot be 0 to view " << TypeIdToString(old_dtype) << " as "
                             << TypeIdToString(new_dtype) << " (different element sizes)";
  }

  std::vector<int64_t> new_shape = old_shape;
  std::vector<int64_t> new_strides;
  int64_t new_offset;

  if (old_element_size > new_element_size) {
    int64_t size_ratio = old_element_size / new_element_size;
    new_strides = DownsizingElementSize(old_strides, size_ratio, old_dtype, new_dtype);
    new_shape[old_shape.size() - 1] *= size_ratio;
    new_offset = size_ratio * old_offset;
  } else {
    int64_t size_ratio = new_element_size / old_element_size;
    if (old_shape.back() % size_ratio != 0) {
      MS_EXCEPTION(ValueError) << "Last dimension must be divisible by " << size_ratio
                               << " to view " << TypeIdToString(old_dtype) << " as "
                               << TypeIdToString(new_dtype) << " (different element sizes), but got "
                               << old_shape.back();
    }

    if (old_offset % size_ratio != 0) {
      MS_EXCEPTION(ValueError) << "Storage offset must be divisible by " << size_ratio
                               << " to view " << TypeIdToString(old_dtype) << " as "
                               << TypeIdToString(new_dtype) << " (different element sizes), but got "
                               << old_offset;
    }

    new_strides = UpsizingElementSize(old_strides, size_ratio, old_dtype, new_dtype);
    new_shape[old_shape.size() - 1] /= size_ratio;
    new_offset = old_offset / size_ratio;
  }
  return {std::make_shared<TensorStorageInfo>(new_shape, new_strides, new_offset, ori_shape, ori_strides,
                                              IsContiguous(new_shape, new_strides)), new_dtype};
}
}  // namespace mindspore::ops
