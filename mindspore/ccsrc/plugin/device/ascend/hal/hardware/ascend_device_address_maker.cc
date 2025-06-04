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
namespace ascend {
DeviceSyncPtr MakeAscendDeviceAddress(TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                      DeviceAddressDeleter &&deleter) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto data_size = SizeOf(shape) * abstract::TypeIdSize(data_type);
  auto device_context = DeviceContextManager::GetInstance().GetOrCreateDeviceContext({"Ascend", device_id});

  auto device_address = device_context->device_res_manager_->CreateDeviceAddress(
    data_ptr, data_size, shape, Format::DEFAULT_FORMAT, data_type, "Ascend", device_id, 0);
  device_address->SetPointerRefCountDeleter(std::move(deleter));
  return device_address;
}

REGISTER_DEVICE_ADDRESS_MAKER(device::DeviceType::kAscend, [](TypeId data_type, const ShapeVector &shape,
                                                              void *data_ptr, DeviceAddressDeleter &&deleter) {
  return MakeAscendDeviceAddress(data_type, shape, data_ptr, std::move(deleter));
});
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
