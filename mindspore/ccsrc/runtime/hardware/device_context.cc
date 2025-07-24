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

#include "runtime/hardware/device_context.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
void *DeviceResManager::AllocateOffloadMemory(size_t size) const { return offloaded_mem_pool_->MallocHost(size); }

void DeviceResManager::FreeOffloadMemory(void *ptr) const { offloaded_mem_pool_->FreeHost(ptr); }

bool DeviceResManager::AllocateMemory(DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(address);
  if (address->GetPtr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected!";
    return false;
  }

  if (stream_id == UINT32_MAX) {
    stream_id = address->stream_id();
  }
  auto device_ptr = AllocateMemory(address->GetSize(), stream_id);
  if (!device_ptr) {
    MS_LOG(WARNING) << "Allocate memory failed for size: " << address->GetSize();
    return false;
  }
  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  static std::string name = "Allocate memory";
  address->IncreaseNewRefCount(name);
  return true;
}

void DeviceResManager::FreeMemory(DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);
  if (address->GetPtr() == nullptr) {
    MS_LOG(EXCEPTION) << "Device ptr is null in device address:" << address << " to release!";
  }

  if (!address->from_mem_pool()) {
    MS_LOG(DEBUG) << "device address:" << address << " ptr:" << address->GetMutablePtr() << " not from pool";
    return;
  }
  MS_LOG(DEBUG) << "Free memory from device address:" << address << " ptr:" << address->GetMutablePtr();
  FreeMemory(address->GetMutablePtr());
  address->set_ptr(nullptr);
}
}  // namespace device
}  // namespace mindspore
