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

#include "view/as_strided_strides_calc.h"
#include <vector>
#include <memory>
#include <map>
#include <functional>

namespace mindspore::ops {
constexpr size_t kAsStridedInputsNum = 4;
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

int64_t ComputeStorageNbytes(const std::vector<int64_t> &shape, const std::vector<int64_t> &stride,
                             int64_t item_bytes) {
  int64_t total_size = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    if (shape[i] == 0) {
      return 0;
    }
    total_size += stride[i] * (shape[i] - 1);
  }
  return total_size * item_bytes;
}

TensorStorageInfoPtrList AsStridedBasicTypeCalc(const PrimitivePtr &prim,
                                                const mindspore::tensor::TensorPtr &input_tensor,
                                                const std::vector<int64_t> &size, const std::vector<int64_t> &stride,
                                                const std::optional<int64_t> &storage_offset_opt) {
  if (size.size() != stride.size()) {
    MS_EXCEPTION(RuntimeError) << "mismatch in length of strides and shape";
  }
  if (std::any_of(size.begin(), size.end(), [](const int64_t &shape_i) { return shape_i < 0; })) {
    MS_EXCEPTION(RuntimeError) << "For primitive[" << prim->name()
                               << "], the component of shape can't be less than 0, but got " << size;
  }
  if (std::any_of(stride.begin(), stride.end(), [](const int &stride_i) { return stride_i < 0; })) {
    MS_EXCEPTION(RuntimeError) << "As_strided: Negative strides are not supported at the moment, but got " << stride;
  }
  int64_t storage_offset = 0;
  if (storage_offset_opt.has_value()) {
    storage_offset = storage_offset_opt.value();
  }
  if (storage_offset < 0) {
    MS_EXCEPTION(RuntimeError) << "As_strided: Invalid storage offset " << storage_offset;
  }

  auto old_tensor_info = GetOldTensorInfo(input_tensor);
  TypeId data_type = input_tensor->data_type();
  int64_t item_size = static_cast<int64_t>(GetDtypeSize(data_type));
  int64_t old_storage_total = std::accumulate(old_tensor_info->old_shape.begin(), old_tensor_info->old_shape.end(),
                                              item_size, std::multiplies<int64_t>());
  int64_t storage_size = ComputeStorageNbytes(size, stride, item_size);
  int64_t storage_offset_size = storage_offset * item_size;
  int64_t new_storage_total = storage_size + storage_offset_size;
  if (new_storage_total > old_storage_total) {
    MS_EXCEPTION(RuntimeError) << "setStorage: sizes " << size << ", strides " << stride << ", storage offset "
                               << storage_offset << ", and itemsize " << item_size << " requiring a storage size of "
                               << new_storage_total << " are out of bounds for storage of size " << old_storage_total;
  }

  // To do check
  auto new_storage_info = std::make_shared<TensorStorageInfo>(size, stride, storage_offset, old_tensor_info->ori_shape,
                                                              old_tensor_info->ori_strides, IsContiguous(size, stride));
  return {new_storage_info};
}

TensorStorageInfoPtrList AsStridedCalc(const PrimitivePtr &prim, const std::vector<ValuePtr> &inputs) {
  if (inputs.size() != kAsStridedInputsNum) {
    return {};
  }
  auto input_tensor = inputs[0]->cast<tensor::TensorPtr>();
  auto shape = GetValue<std::vector<int64_t>>(inputs[1]);
  auto stride = GetValue<std::vector<int64_t>>(inputs[2]);
  auto storage_offset = GetValue<int64_t>(inputs[3]);
  return AsStridedBasicTypeCalc(prim, input_tensor, shape, stride, storage_offset);
}
}  // namespace mindspore::ops
