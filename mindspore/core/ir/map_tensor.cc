/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ir/map_tensor.h"
#include <vector>
#include <algorithm>
#include "ir/tensor_new.h"
#include "abstract/abstract_value.h"
#include "ir/tensor.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils_secure.h"

namespace mindspore {
namespace tensor {
using tensor::Tensor;
using tensor::TensorPtr;
constexpr size_t kKeyTensorIndex = 0;
constexpr size_t kValueTensorIndex = 1;
constexpr size_t kStatusTensorIndex = 2;
constexpr size_t kExportTensorNum = 3;

static ShapeVector ConcatShape(const ShapeVector &a, const ShapeVector &b) {
  ShapeVector result_shape = a;
  (void)result_shape.insert(result_shape.end(), b.cbegin(), b.cend());
  return result_shape;
}

MapTensor::MapTensor(TypeId key_dtype, TypeId value_dtype, const ShapeVector &value_shape,
                     const ValuePtr &default_value, const ValuePtr &permit_filter_value,
                     const ValuePtr &evict_filter_value)
    : key_dtype_(key_dtype), default_value_(default_value) {
  data_type_ = value_dtype;
  value_shape_ = value_shape;
  key_shape_ = {abstract::Shape::kShapeDimAny};
  shape_ = {abstract::Shape::kShapeDimAny};
  (void)shape_.insert(shape_.cend(), value_shape.cbegin(), value_shape.cend());
  size_ = shape_[0];
  ShapeVector key_shape = {abstract::Shape::kShapeDimAny};
  key_tensor_ = tensor::from_spec(key_dtype, key_shape, device::DeviceType::kCPU);
  value_tensor_ = tensor::from_spec(value_dtype, shape_, device::DeviceType::kCPU);
  status_tensor_ = tensor::from_spec(kNumberTypeInt, key_shape, device::DeviceType::kCPU);
  permit_filter_value_ = (permit_filter_value == nullptr) ? std::make_shared<Int64Imm>(1) : permit_filter_value;
  evict_filter_value_ = (evict_filter_value == nullptr) ? std::make_shared<Int64Imm>(INT64_MAX) : evict_filter_value;
}

std::size_t MapTensor::hash() const { return static_cast<std::size_t>(tid()); }

bool MapTensor::operator==(const MapTensor &other) const { return this == &other; }

abstract::AbstractBasePtr MapTensor::ToAbstract() {
  if (param_info_ != nullptr) {
    // For parameter, a broaden abstract is created with ref_key set.
    ValuePtr ref_key = std::make_shared<RefKey>(param_info_->name());
    return std::make_shared<abstract::AbstractMapTensor>(shared_from_base<MapTensor>(), ref_key);
  } else {
    // For value, an abstract is created with value set.
    return std::make_shared<abstract::AbstractMapTensor>(shared_from_base<MapTensor>());
  }
}

std::string MapTensor::ToString() const {
  auto key_dtype = KeyDtype();
  auto value_dtype = ValueDtype();
  return "MapTensor(key_dtype=" + (key_dtype == nullptr ? "<null>" : key_dtype->ToString()) +
         ", value_dtype=" + (value_dtype == nullptr ? "<null>" : value_dtype->ToString()) +
         ", value_shape=" + tensor::ShapeToString(value_shape()) +
         ", default_value=" + (default_value_ == nullptr ? "<null>" : default_value_->ToString()) +
         ", permit_filter=" + (permit_filter_value_ == nullptr ? "<null>" : permit_filter_value_->ToString()) +
         ", evict_filter=" + (evict_filter_value_ == nullptr ? "<null>" : evict_filter_value_->ToString()) + ")";
}

void MapTensor::Update(const MapTensor::ExportData &data) {
  MS_EXCEPTION_IF_NULL(data.key_tensor);
  MS_EXCEPTION_IF_NULL(data.value_tensor);
  MS_EXCEPTION_IF_NULL(data.status_tensor);
  key_tensor_ = data.key_tensor;
  value_tensor_ = data.value_tensor;
  status_tensor_ = data.status_tensor;
}

void MapTensor::TransExportDataToTensor(const HashTableExportData &export_data) const {
  if (export_data.size() != kExportTensorNum) {
    MS_LOG(EXCEPTION) << "Invalid MapTensor export data.";
  }

  auto keys = export_data.at(kKeyTensorIndex);
  auto values = export_data.at(kValueTensorIndex);
  auto statuses = export_data.at(kStatusTensorIndex);
  MS_EXCEPTION_IF_NULL(keys);
  MS_EXCEPTION_IF_NULL(values);
  MS_EXCEPTION_IF_NULL(statuses);

  // The key tensor.
  auto keys_length = keys->size();
  auto keys_num = keys_length / abstract::TypeIdSize(key_dtype());
  ShapeVector key_tensor_shape{SizeToLong(keys_num)};
  auto tensor_key = key_tensor();
  MS_EXCEPTION_IF_NULL(tensor_key);
  (void)tensor_key->set_shape(key_tensor_shape);
  if (keys_length > 0) {
    auto ret = memcpy_s(tensor_key->data_c(), tensor_key->Size(), keys->data(), keys_length);
    if (ret != EOK) {
      MS_LOG(INTERNAL_EXCEPTION) << "Memcpy for key tensor failed, errno[" << ret << "]";
    }
  }

  // The value tensor.
  auto values_length = values->size();
  auto element_length = LongToSize(abstract::ShapeSize(value_shape())) * abstract::TypeIdSize(value_dtype());
  MS_EXCEPTION_IF_ZERO("element_length", element_length);
  auto values_num = values_length / element_length;
  ShapeVector value_tensor_shape{SizeToLong(values_num)};
  (void)std::copy(value_shape().cbegin(), value_shape().cend(), std::back_inserter(value_tensor_shape));
  auto tensor_value = value_tensor();
  MS_EXCEPTION_IF_NULL(tensor_value);
  (void)tensor_value->set_shape(value_tensor_shape);
  if (values_length > 0) {
    auto ret = memcpy_s(tensor_value->data_c(), tensor_value->Size(), values->data(), values_length);
    if (ret != EOK) {
      MS_LOG(INTERNAL_EXCEPTION) << "Memcpy for value tensor failed, errno[" << ret << "]";
    }
  }

  // The status tensor
  auto statuses_length = statuses->size();
  auto statuses_num = statuses_length / abstract::TypeIdSize(kNumberTypeInt);
  // The status tensor shape is same as the shape of key tensor.
  if (statuses_num != keys_num) {
    MS_LOG(INTERNAL_EXCEPTION) << "Invalid export data: keys num: " << keys_num << ", statuses num: " << statuses_num;
  }
  ShapeVector status_tensor_shape{SizeToLong(statuses_num)};
  auto tensor_status = status_tensor();
  MS_EXCEPTION_IF_NULL(tensor_status);
  (void)tensor_status->set_shape(status_tensor_shape);
  if (statuses_length > 0) {
    auto ret = memcpy_s(tensor_status->data_c(), tensor_status->Size(), statuses->data(), statuses_length);
    if (ret != EOK) {
      MS_LOG(INTERNAL_EXCEPTION) << "Memcpy for status tensor failed, errno[" << ret << "]";
    }
  }
}

MapTensor::ExportData MapTensor::ExportDataFromDevice(const DeviceAddressPtr &device_sync, bool incremental,
                                                      bool *last_slice) const {
  MS_LOG(EXCEPTION) << "Call deprecated interface ExportDataFromDevice.";
}

// If the data on the host side is valid, the data on the host side will be exported.
bool MapTensor::CheckData() const {
  // check key
  auto tensor_key = key_tensor();
  MS_EXCEPTION_IF_NULL(tensor_key);
  if (tensor_key->shape().size() != 1 || tensor_key->shape()[0] < 1) {
    MS_LOG(WARNING) << "Invalid key tensor shape: " << tensor::ShapeToString(tensor_key->shape());
    return false;
  }
  // check value
  bool check_value =
    std::any_of(value_shape().cbegin(), value_shape().cend(), [](const ShapeValueDType &shape) { return shape < 1; });
  if (check_value) {
    MS_LOG(WARNING) << "Invalid value tensor shape: " << tensor::ShapeToString(value_shape());
    return false;
  }
  // check status
  auto tensor_status = status_tensor();
  MS_EXCEPTION_IF_NULL(tensor_status);
  if (tensor_status->shape().size() != 1 || tensor_status->shape()[0] < 1) {
    MS_LOG(WARNING) << "Invalid status tensor shape: " << tensor::ShapeToString(tensor_status->shape());
    return false;
  }
  return true;
}

MapTensor::ExportData MapTensor::Export(bool incremental) const {
  MS_LOG(DEBUG) << (incremental ? "Incremental" : "Full") << " export MapTensor";

  // Check device
  DeviceAddressPtr device_sync = device_address();
  if (device_sync != nullptr) {
    return ExportDataFromDevice(device_sync, incremental);
  }
  if (CheckData()) {
    return {key_tensor(), value_tensor(), status_tensor()};
  }
  // Note: this is fake implementation.
  ShapeVector key_shape = {1};
  ShapeVector values_shape = ConcatShape(ShapeVector{1}, value_shape());
  auto key_tensor = tensor::from_spec(key_dtype(), key_shape, device::DeviceType::kCPU);
  auto value_tensor = tensor::from_spec(value_dtype(), values_shape, device::DeviceType::kCPU);
  auto status_tensor = tensor::from_spec(kNumberTypeInt, key_shape, device::DeviceType::kCPU);
  return {key_tensor, value_tensor, status_tensor};
}

MapTensor::ExportData MapTensor::ExportSlice(bool incremental, bool *last_slice) const {
  MS_EXCEPTION_IF_NULL(last_slice);
  DeviceAddressPtr device_sync = device_address();
  MS_EXCEPTION_IF_NULL(device_sync);
  return ExportDataFromDevice(device_sync, incremental, last_slice);
}
}  // namespace tensor
}  // namespace mindspore
