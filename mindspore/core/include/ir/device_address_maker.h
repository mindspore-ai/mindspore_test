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

#ifndef MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_DEVICE_ADDRESS_MAKER_H_
#define MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_DEVICE_ADDRESS_MAKER_H_

#include <functional>
#include <memory>
#include <utility>
#include "mindapi/base/type_id.h"
#include "mindapi/base/shape_vector.h"
#include "ir/device_type.h"
#include "ir/tensor_data.h"

namespace mindspore {
class DeviceSync;
using DeviceSyncPtr = std::shared_ptr<DeviceSync>;

using DeviceAddressDeleter = std::function<void(void *)>;
using DeviceAddressMakerFunc =
  std::function<DeviceSyncPtr(TypeId, const ShapeVector &, void *data_ptr, DeviceAddressDeleter &&)>;
MS_CORE_API void SetDeviceAddressMaker(device::DeviceType device_type, DeviceAddressMakerFunc &&func);
MS_CORE_API DeviceAddressMakerFunc GetDeviceAddressMaker(device::DeviceType device_target);

template <device::DeviceType t>
struct MS_CORE_API DeviceAddressMakerRegister {
  explicit DeviceAddressMakerRegister(DeviceAddressMakerFunc &&maker) { SetDeviceAddressMaker(t, std::move(maker)); }
};

#define REGISTER_DEVICE_ADDRESS_MAKER(t, f)                 \
  namespace {                                               \
  static DeviceAddressMakerRegister<t> g_maker_register(f); \
  }

class MS_CORE_API DeviceAddressMaker {
 public:
  DeviceAddressMaker(void *data_ptr, TypeId data_type, const ShapeVector &shape)
      : data_ptr_(data_ptr), data_type_(data_type), shape_(shape) {}

  DeviceSyncPtr make_device_address();

  DeviceAddressMaker &set_deleter(std::function<void(void *)> &&deleter);

  DeviceAddressMaker &set_maker(DeviceAddressMakerFunc &&maker);

 private:
  void *data_ptr_;
  TypeId data_type_;
  const ShapeVector &shape_;
  std::function<void(void *)> deleter_;
  DeviceAddressMakerFunc maker_;
};

MS_CORE_API DeviceSyncPtr MakeDeviceAddress(TypeId data_type, const ShapeVector &shape, bool init = false,
                                            device::DeviceType device_type = device::DeviceType::kCPU);
MS_CORE_API DeviceSyncPtr MakeDeviceAddress(TypeId data_type, const ShapeVector &shape,
                                            const tensor::TensorDataPtr &tensor_data,
                                            device::DeviceType device_type = device::DeviceType::kCPU);

}  // namespace mindspore

#endif  // MINDSPORE_MINDSPORE_CORE_INCLUDE_IR_DEVICE_ADDRESS_MAKER_H_
