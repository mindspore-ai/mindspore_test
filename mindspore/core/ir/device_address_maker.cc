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

#include "ir/device_address_maker.h"
#include "ir/device_sync.h"
#include "ir/tensor_data.h"

namespace mindspore {
constexpr int kMaxDeviceNum = 5;
DeviceAddressMakerFunc g_device_address_maker[kMaxDeviceNum];
DeviceSyncPtr DeviceAddressMaker::make_device_address() {
  auto device_sync = maker_(data_type_, shape_, data_ptr_, std::move(deleter_));
  device_sync->set_deleter(deleter_);
  return device_sync;
}

DeviceAddressMaker &DeviceAddressMaker::set_deleter(std::function<void(void *)> &&deleter) {
  deleter_ = std::move(deleter);
  return *this;
}

DeviceAddressMaker &DeviceAddressMaker::set_maker(DeviceAddressMakerFunc &&maker) {
  maker_ = std::move(maker);
  return *this;
}

void SetDeviceAddressMaker(device::DeviceType device_type, DeviceAddressMakerFunc &&func) {
  g_device_address_maker[static_cast<int>(device_type)] = func;
}

DeviceAddressMakerFunc GetDeviceAddressMaker(device::DeviceType device_target) {
  const auto &maker = g_device_address_maker[static_cast<int>(device_target)];
  MS_EXCEPTION_IF_NULL(maker);
  return maker;
}

DeviceSyncPtr MakeDeviceAddress(TypeId data_type, const ShapeVector &shape, bool init, device::DeviceType device_type) {
  // todo: set allocator
  auto tensor_data = tensor::MakeTensorData(data_type, shape);
  return DeviceAddressMaker(init ? tensor_data->data() : tensor_data->const_data(), data_type, shape)
    .set_deleter([tensor_data](void *) {})
    .set_maker(GetDeviceAddressMaker(device_type))
    .make_device_address();
}

DeviceSyncPtr MakeDeviceAddress(TypeId data_type, const ShapeVector &shape, const tensor::TensorDataPtr &tensor_data,
                                device::DeviceType device_type) {
  // Just GET data ptr of tensor_data and don't init the data.
  return DeviceAddressMaker(tensor_data->const_data(), data_type, shape)
    .set_deleter([tensor_data](void *) {})
    .set_maker(GetDeviceAddressMaker(device_type))
    .make_device_address();
}
}  // namespace mindspore
