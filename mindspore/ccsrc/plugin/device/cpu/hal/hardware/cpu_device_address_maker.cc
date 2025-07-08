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

#include "ir/device_type.h"
#include "common/device_address.h"
#include "ir/device_address_maker.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace device {
namespace cpu {
DeviceSyncPtr MakeCPUDeviceAddress(TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                   DeviceAddressDeleter &&deleter) {
  auto data_size = SizeOf(shape) * abstract::TypeIdSize(data_type);
  auto device_address =
    std::make_shared<DeviceAddress>(data_ptr, data_size, shape, Format::DEFAULT_FORMAT, data_type, "CPU", 0, 0);
  if (deleter != nullptr) {
    device_address->SetPointerRefCountDeleter(std::move(deleter));
  }
  return device_address;
}

REGISTER_DEVICE_ADDRESS_MAKER(device::DeviceType::kCPU, [](TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                                           DeviceAddressDeleter &&deleter) {
  return MakeCPUDeviceAddress(data_type, shape, data_ptr, std::move(deleter));
});
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
