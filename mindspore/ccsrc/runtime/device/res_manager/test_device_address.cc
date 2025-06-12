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

#include "runtime/device/res_manager/test_device_address.h"
#include "ir/device_address_maker.h"
#include "runtime/device/res_manager/utils/convert_tensor_utils.h"

namespace mindspore {
namespace runtime {
namespace test {
namespace {
DeviceSyncPtr MakeTestDeviceAddress(TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                    DeviceAddressDeleter &&deleter) {
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto data_size = SizeOf(shape) * abstract::TypeIdSize(data_type);
  auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({"CPU", device_id});
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  auto device_address = device_context->device_res_manager_->CreateDeviceAddress(
    data_ptr, data_size, shape, Format::DEFAULT_FORMAT, data_type, "CPU", device_id, 0);
  device_address->SetPointerRefCountDeleter(std::move(deleter));
  return device_address;
}

const char device_name[] = "CPU";
MS_REGISTER_DEVICE(device_name, TestDeviceContext);
REGISTER_DEVICE_ADDRESS_MAKER(device::DeviceType::kCPU, [](TypeId data_type, const ShapeVector &shape, void *data_ptr,
                                                           DeviceAddressDeleter &&deleter) {
  return MakeTestDeviceAddress(data_type, shape, data_ptr, std::move(deleter));
});
}  // namespace

bool TestResManager::SyncCopy(const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync,
                              size_t stream_id) const {
  return AsyncCopy(dst_device_sync, src_device_sync, stream_id, false);
}

bool TestResManager::AsyncCopy(const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync,
                               size_t stream_id, bool) const {
  const auto &dst_device_address = dynamic_cast<const TestDeviceAddress *>(dst_device_sync.get());
  const auto &src_device_address = dynamic_cast<const TestDeviceAddress *>(src_device_sync.get());
  MS_EXCEPTION_IF_NULL(dst_device_address);
  MS_EXCEPTION_IF_NULL(src_device_address);
  if (dst_device_address->GetSize() == 0 || src_device_address->GetSize() == 0) {
    MS_LOG(INFO) << "No need sync for dst device address: " << dst_device_address->PrintInfo()
                 << " and src device address: " << src_device_address->PrintInfo();
    return true;
  }

  if (dst_device_address->format() != src_device_address->format()) {
    MS_LOG(ERROR) << "Format is different, src(format:" << src_device_address->format()
                  << "), dst(format:" << dst_device_address->format() << ") for device address:" << dst_device_address;
    return false;
  }
  auto dst_ptr = dst_device_address->GetMutablePtr();
  auto src_ptr = src_device_address->GetMutablePtr();
  MS_EXCEPTION_IF_NULL(src_device_address->GetMutablePtr());
  MS_EXCEPTION_IF_NULL(dst_device_address->GetMutablePtr());
  if (dst_ptr == src_ptr) {
    MS_LOG(DEBUG) << "host_ptr is equal to device ptr, request ignored.";
    return true;
  }
  auto dst_type_id = dst_device_address->type_id();
  auto src_type_id = src_device_address->type_id();

  if (src_type_id == dst_type_id) {
    if (src_device_address->GetSize() > dst_device_address->GetSize()) {
      MS_LOG(WARNING) << "Please check whether need sync data, src size: " << src_device_address->GetSize()
                      << ", dst size: " << dst_device_address->GetSize();
      return true;
    }
    auto ret_code = memcpy_s(dst_ptr, src_device_address->GetSize(), src_ptr, src_device_address->GetSize());
    // Return ERANGE when the copy size is larger than SECUREC_MEM_MAX_LEN.
    if (ret_code == ERANGE) {
      device::ConvertSameType(dst_device_address->GetMutablePtr(), src_device_address->GetMutablePtr(),
                              dst_device_address->GetSize(), src_type_id);
    } else if (ret_code != EOK) {
      MS_LOG(ERROR) << "Failed to copy tensor from device address:" << src_device_address->PrintInfo()
                    << " to :" << dst_device_address->PrintInfo();
      return false;
    } else {
      return true;
    }
  }

  if (dst_type_id == kNumberTypeFloat16 && src_type_id == kNumberTypeFloat32) {
    device::FloatToHalf(dst_ptr, src_ptr, dst_device_address->GetSize() >> 1);
  } else if (dst_type_id == kNumberTypeFloat64 && src_type_id == kNumberTypeFloat32) {
    device::FloatToDouble(dst_ptr, src_ptr, dst_device_address->GetSize() / sizeof(double));
  } else if (dst_type_id == kNumberTypeFloat32 && src_type_id == kNumberTypeFloat64) {
    device::DoubleToFloat(dst_ptr, src_ptr, dst_device_address->GetSize() >> 2);
  } else if (dst_type_id == kNumberTypeInt16 && src_type_id == kNumberTypeInt32) {
    device::IntToShort(dst_ptr, src_ptr, dst_device_address->GetSize() >> 1);
  } else if (dst_type_id == kNumberTypeInt64 && src_type_id == kNumberTypeInt32) {
    device::IntToLong(dst_ptr, src_ptr, dst_device_address->GetSize() / sizeof(int64_t));
  } else {
    MS_LOG(ERROR) << "Types not match. src type: " << TypeIdLabel(src_type_id)
                  << ", dst type: " << TypeIdLabel(dst_type_id) << " device_address:" << dst_device_address << " !";
    return false;
  }
  return true;
}

MS_REGISTER_HAL_COPY_FUNC(
  DeviceType::kCPU, ([](const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync, size_t stream_id) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::ResKey res_key{DeviceType::kCPU, device_id};
    auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);
    return res_manager->SyncCopy(dst_device_sync, src_device_sync, stream_id);
  }),
  ([](const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync, size_t stream_id, bool) {
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    auto device_id = context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    device::ResKey res_key{DeviceType::kCPU, device_id};
    auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);
    return res_manager->SyncCopy(dst_device_sync, src_device_sync, stream_id);
  }));

MS_REGISTER_HAL_RES_MANAGER(kCPUDevice, DeviceType::kCPU, TestResManager);
}  // namespace test
}  // namespace runtime
}  // namespace mindspore
