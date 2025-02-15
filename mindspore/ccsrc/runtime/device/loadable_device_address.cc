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

#include "runtime/device/loadable_device_address.h"
#include "include/common/debug/common.h"
#include "include/common/utils/offload_context.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace device {
namespace {
constexpr size_t kFileAlignSize = 512;
constexpr char kSwapFileSuffix[] = ".data";
}  // namespace

bool LoadableDeviceAddress::MoveTo(mindspore::device::StorageType dst, bool async, size_t stream_id) {
  bool ret = Wait();
  if (!ret) {
    MS_LOG(WARNING) << "Wait swapping DeviceAddress failed. Status: " << status_;
    return false;
  }
  if (status_ == DeviceAddressStatus::kInDevice && GetDevicePtr() == nullptr) {
    MS_LOG(INFO) << "Skip move empty device address.";
    return true;
  }
  if (dst == StorageType::kDevice) {
    if (!MoveToDevice(async, stream_id)) {
      MS_LOG(WARNING) << "Move data to device failed.";
      return false;
    }
  } else if (dst == StorageType::kHost) {
    if (!MoveToHost(async, stream_id)) {
      MS_LOG(WARNING) << "Move data to host failed.";
      return false;
    }
  } else if (dst == StorageType::kFile) {
    if (!MoveToFile(async, stream_id)) {
      MS_LOG(WARNING) << "Move data to file failed.";
      return false;
    }
  }
  return true;
}

bool LoadableDeviceAddress::MoveToHost(bool async, size_t stream_id) const {
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  if (loadable_mem_ == nullptr) {
    loadable_mem_ = std::make_unique<LoadableMember>();
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (loadable_mem_->storage_info_.host_ptr_ == nullptr || loadable_mem_->storage_info_.host_ptr_mutable_) {
    loadable_mem_->storage_info_.host_ptr_ = swap_manager->AllocHostMemory(GetFileAlignSize());
    if (loadable_mem_->storage_info_.host_ptr_ == nullptr) {
      MS_LOG(WARNING) << "Allocating host memory failed, size: " << GetSize();
      return false;
    }
  }
  if (status_ == DeviceAddressStatus::kInFile) {
    if (!CopyFileToHost(loadable_mem_->storage_info_.host_ptr_, loadable_mem_->storage_info_.file_name_, GetSize(),
                        async)) {
      MS_LOG(WARNING) << "Copy data from file to host failed.";
      return false;
    }
    if (async) {
      swap_manager->AddSwappingTensor(this);
      status_ = DeviceAddressStatus::kInFileToHost;
    } else {
      if (loadable_mem_->storage_info_.file_name_mutable_) {
        (void)swap_manager->DeleteFile(loadable_mem_->storage_info_.file_name_);
        loadable_mem_->storage_info_.file_name_ = "";
      }
      status_ = DeviceAddressStatus::kInHost;
    }
  } else {
    if (!CopyDeviceToHost(loadable_mem_->storage_info_.host_ptr_, GetDevicePtr(), GetSize(), async, stream_id)) {
      MS_LOG(WARNING) << "Copy data from device to host failed.";
      return false;
    }
    if (async) {
      swap_manager->AddSwappingTensor(this);
      status_ = DeviceAddressStatus::kInDeviceToHost;
    } else {
      swap_manager->FreeDeviceMemory(GetDevicePtr());
      SetDevicePtr(nullptr);
      status_ = DeviceAddressStatus::kInHost;
    }
  }
  return true;
}

bool LoadableDeviceAddress::MoveToDevice(bool async, size_t stream_id) const {
  if (status_ == DeviceAddressStatus::kInDevice) {
    return true;
  }
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  MS_EXCEPTION_IF_NULL(loadable_mem_);
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (status_ == DeviceAddressStatus::kInFile) {
#if defined(RT_MEMORY_P2PDMA)
    if (GetDevicePtr() == nullptr) {
      SetDevicePtr(swap_manager->AllocDeviceMemory(GetSize(), stream_id));
    }
    MS_EXCEPTION_IF_NULL(GetDevicePtr());
    if (FileToDeviceDirectly(GetDevicePtr(), GetSize(), loadable_mem_->storage_info_.file_name_, stream_id)) {
      if (loadable_mem_->storage_info_.file_name_mutable_ && !loadable_mem_->storage_info_.file_name_.empty()) {
        (void)swap_manager->DeleteFile(loadable_mem_->storage_info_.file_name_);
        loadable_mem_->storage_info_.file_name_ = "";
      }
      if (loadable_mem_->storage_info_.host_ptr_mutable_) {
        swap_manager->FreeHostMemory(loadable_mem_->storage_info_.host_ptr_);
        loadable_mem_->storage_info_.host_ptr_ = nullptr;
      }
      status_ = DeviceAddressStatus::kInDevice;
      return true;
    }
#endif
    if (!MoveToHost(false, stream_id)) {
      return false;
    }
  }
  if (GetDevicePtr() == nullptr) {
    SetDevicePtr(swap_manager->AllocDeviceMemory(GetSize(), stream_id));
    if (GetDevicePtr() == nullptr) {
      MS_LOG(WARNING) << "Allocating device memory failed, size: " << GetSize();
      return false;
    }
  }
  if (!CopyHostToDevice(GetDevicePtr(), loadable_mem_->storage_info_.host_ptr_, GetSize(), async, stream_id)) {
    MS_LOG(WARNING) << "Copy data from host to device failed.";
    return false;
  }
  if (async) {
    swap_manager->AddSwappingTensor(this);
    status_ = DeviceAddressStatus::kInHostToDevice;
  } else {
    if (loadable_mem_->storage_info_.host_ptr_mutable_) {
      swap_manager->FreeHostMemory(loadable_mem_->storage_info_.host_ptr_);
      loadable_mem_->storage_info_.host_ptr_ = nullptr;
    }

    status_ = DeviceAddressStatus::kInDevice;
  }
  return true;
}

bool LoadableDeviceAddress::MoveToFile(bool async, size_t stream_id) const {
  if (status_ == DeviceAddressStatus::kInFile) {
    return true;
  }
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  if (loadable_mem_ == nullptr) {
    loadable_mem_ = std::make_unique<LoadableMember>();
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (status_ == DeviceAddressStatus::kInDevice) {
#if defined(RT_MEMORY_P2PDMA)
    if (loadable_mem_->storage_info_.file_name_.empty() || loadable_mem_->storage_info_.file_name_mutable_) {
      loadable_mem_->storage_info_.file_name_ = GetSwapFileName();
    }
    if (DeviceToFileDirectly(GetDevicePtr(), GetSize(), loadable_mem_->storage_info_.file_name_, stream_id)) {
      status_ = DeviceAddressStatus::kInFile;
      if (GetDevicePtr() != nullptr) {
        swap_manager->FreeDeviceMemory(GetDevicePtr());
        SetDevicePtr(nullptr);
      }
      if (loadable_mem_->storage_info_.host_ptr_ != nullptr) {
        swap_manager->FreeHostMemory(loadable_mem_->storage_info_.host_ptr_);
        loadable_mem_->storage_info_.host_ptr_ = nullptr;
      }
      return true;
    }
#endif
    if (!MoveToHost(false, stream_id)) {
      return false;
    }
  }
  if (loadable_mem_->storage_info_.file_name_.empty() || loadable_mem_->storage_info_.file_name_mutable_) {
    loadable_mem_->storage_info_.file_name_ = GetSwapFileName();
    if (!swap_manager->CreateFile(loadable_mem_->storage_info_.file_name_, GetFileAlignSize())) {
      MS_LOG(WARNING) << "Create file for swapping failed.";
      return false;
    }
  }
  if (!CopyHostToFile(loadable_mem_->storage_info_.file_name_, loadable_mem_->storage_info_.host_ptr_, GetSize(),
                      async)) {
    MS_LOG(WARNING) << "Copy data from host to file failed.";
    return false;
  }
  if (async) {
    swap_manager->AddSwappingTensor(this);
    status_ = DeviceAddressStatus::kInHostToFile;
  } else {
    if (loadable_mem_->storage_info_.host_ptr_mutable_) {
      swap_manager->FreeHostMemory(loadable_mem_->storage_info_.host_ptr_);
      loadable_mem_->storage_info_.host_ptr_ = nullptr;
    }
    status_ = DeviceAddressStatus::kInFile;
  }
  return true;
}

bool LoadableDeviceAddress::CopyHostToFile(const std::string &dst, const void *src, size_t size, bool async) const {
  MS_EXCEPTION_IF_NULL(src);
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  AsyncIOToken token;
  bool ret = swap_manager->HostMemoryToFile(dst, src, size, async, &token);
  if (!ret) {
    MS_LOG(WARNING) << "Write data from ddr to file[" << dst << "] failed.";
    return ret;
  }
  if (async) {
    MS_EXCEPTION_IF_NULL(loadable_mem_);
    loadable_mem_->swap_event_.aio_token_ = token;
  }
  return ret;
}

bool LoadableDeviceAddress::CopyFileToHost(void *dst, const std::string &src, size_t size, bool async) const {
  MS_EXCEPTION_IF_NULL(dst);
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  AsyncIOToken token;
  bool ret = swap_manager->FileToHostMemory(dst, src, size, async, &token);
  if (!ret) {
    MS_LOG(WARNING) << "Read data from file[" << src << "] to ddr failed.";
    return ret;
  }
  if (async) {
    MS_EXCEPTION_IF_NULL(loadable_mem_);
    loadable_mem_->swap_event_.aio_token_ = token;
  }
  return true;
}

void LoadableDeviceAddress::ReleaseResource() {
  if (loadable_mem_ == nullptr || status_ == DeviceAddressStatus::kInDevice) {
    return;
  }

  const bool need_delete_file =
    !loadable_mem_->storage_info_.file_name_.empty() && loadable_mem_->storage_info_.file_name_mutable_;
  const bool need_free_host =
    loadable_mem_->storage_info_.host_ptr_ != nullptr && loadable_mem_->storage_info_.host_ptr_mutable_;
  if (need_delete_file || need_free_host) {
    auto device_context = GetDeviceContext();
    MS_EXCEPTION_IF_NULL(device_context);
    const auto swap_manager = device_context->device_res_manager_->swap_manager();
    MS_EXCEPTION_IF_NULL(swap_manager);
    if (need_delete_file) {
      (void)swap_manager->DeleteFile(loadable_mem_->storage_info_.file_name_);
    }
    if (need_free_host) {
      swap_manager->FreeHostMemory(loadable_mem_->storage_info_.host_ptr_);
    }
  }
  loadable_mem_ = nullptr;
}

size_t LoadableDeviceAddress::GetFileAlignSize() const {
  return (GetSize() + kFileAlignSize - 1) / kFileAlignSize * kFileAlignSize;
}

std::string LoadableDeviceAddress::GetSwapFileName() const {
  static size_t swap_file_index = 0;
  std::string file_dir;
  const auto &offload_context = OffloadContext::GetInstance();
  if (offload_context != nullptr) {
    const auto real_dir = FileUtils::GetRealPath(offload_context->offload_path().c_str());
    if (!real_dir.has_value()) {
      MS_LOG(EXCEPTION) << "Invalid offload path[" << offload_context->offload_path()
                        << "]. Please check offload_path configuration.";
    }
    file_dir = real_dir.value() + "/";
  }
  return file_dir + std::to_string(device_id()) + "_" + std::to_string(swap_file_index++) + "_" +
         std::to_string(Common::GetTimeStamp()) + kSwapFileSuffix;
}

void LoadableDeviceAddress::SetStorageInfo(const StorageInfo &storage_info) {
  if (loadable_mem_ == nullptr) {
    loadable_mem_ = std::make_unique<LoadableMember>();
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  loadable_mem_->storage_info_ = storage_info;
  if (loadable_mem_->storage_info_.host_ptr_ != nullptr) {
    status_ = DeviceAddressStatus::kInHost;
    loadable_mem_->storage_info_.host_ptr_mutable_ = false;
  } else if (!loadable_mem_->storage_info_.file_name_.empty()) {
    status_ = DeviceAddressStatus::kInFile;
    loadable_mem_->storage_info_.file_name_mutable_ = false;
  } else {
    status_ = DeviceAddressStatus::kInDevice;
  }
}

StorageInfo LoadableDeviceAddress::GetStorageInfo() const {
  if (loadable_mem_ == nullptr) {
    loadable_mem_ = std::make_unique<LoadableMember>();
  }
  return loadable_mem_->storage_info_;
}

void LoadableDeviceAddress::Swap(mindspore::device::DeviceAddress *other) {
  DeviceAddress::Swap(other);
  if (other == this) {
    return;
  }
  auto loadable_device_address = reinterpret_cast<LoadableDeviceAddress *>(other);
  if (loadable_device_address != nullptr) {
    if (loadable_mem_ == nullptr) {
      loadable_mem_ = std::make_unique<LoadableMember>();
    }
    if (loadable_device_address->loadable_mem_ == nullptr) {
      loadable_device_address->loadable_mem_ = std::make_unique<LoadableMember>();
    }
    loadable_device_address->loadable_mem_->storage_info_ = loadable_mem_->storage_info_;
    loadable_device_address->status_ = status_;
    loadable_mem_->storage_info_.host_ptr_ = nullptr;
    loadable_mem_->storage_info_.file_name_ = "";
    loadable_mem_->storage_info_.host_ptr_mutable_ = true;
    loadable_mem_->storage_info_.file_name_mutable_ = true;
    status_ = DeviceAddressStatus::kInDevice;
  }
}

bool LoadableDeviceAddress::Wait() const {
  if (loadable_mem_ == nullptr || !loadable_mem_->swap_event_.NeedWait()) {
    return true;
  }
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  const auto device_context = GetDeviceContext();
  MS_EXCEPTION_IF_NULL(device_context);
  const auto swap_manager = device_context->device_res_manager_->swap_manager();
  MS_EXCEPTION_IF_NULL(swap_manager);
  if (loadable_mem_->swap_event_.device_event_ != nullptr && loadable_mem_->swap_event_.device_event_->NeedWait()) {
    loadable_mem_->swap_event_.device_event_->WaitEvent();
  } else if (loadable_mem_->swap_event_.aio_token_ != kInvalidAsyncIOToken) {
    if (!swap_manager->WaitAsyncIO(loadable_mem_->swap_event_.aio_token_)) {
      MS_LOG(WARNING) << "Wait aio failed.";
      return false;
    }
  } else {
    MS_LOG(WARNING) << "Device address is in moving, but no valid swap event can be found.";
  }
  if (status_ == DeviceAddressStatus::kInFileToHost) {
    if (loadable_mem_->storage_info_.file_name_mutable_) {
      (void)swap_manager->DeleteFile(loadable_mem_->storage_info_.file_name_);
      loadable_mem_->storage_info_.file_name_ = "";
    }
    status_ = DeviceAddressStatus::kInHost;
  } else if (status_ == DeviceAddressStatus::kInDeviceToHost) {
    swap_manager->FreeDeviceMemory(GetDevicePtr());
    status_ = DeviceAddressStatus::kInHost;
  } else {
    if (loadable_mem_->storage_info_.host_ptr_mutable_) {
      swap_manager->FreeHostMemory(loadable_mem_->storage_info_.host_ptr_);
      loadable_mem_->storage_info_.host_ptr_ = nullptr;
    }
    if (status_ == DeviceAddressStatus::kInHostToDevice) {
      status_ = DeviceAddressStatus::kInHost;
    } else {
      status_ = DeviceAddressStatus::kInFile;
    }
  }
  return true;
}

// Return whether DeviceAddress has a valid ptr.
bool LoadableDeviceAddress::IsPtrValid() const {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (GetDevicePtr() != nullptr) {
    return true;
  }
  const auto &hete_info = kernel_tensor_ == nullptr ? nullptr : kernel_tensor_->heterogeneous_info();
  if (hete_info != nullptr) {
    if (hete_info->host_ptr_ != nullptr || !hete_info->file_name_.empty()) {
      return true;
    }
  }
  if (loadable_mem_ != nullptr) {
    if (loadable_mem_->storage_info_.host_ptr_ != nullptr || !loadable_mem_->storage_info_.file_name_.empty()) {
      return true;
    }
  }
  return false;
}

// Load first if data is offloaded and return the device ptr.
void *LoadableDeviceAddress::GetValidPtr(size_t stream_id) {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (!MoveToDevice(false)) {
    MS_LOG(ERROR) << "Move data to device failed.";
    return nullptr;
  }
  return DeviceAddress::GetValidPtr(stream_id);
}
}  // namespace device
}  // namespace mindspore
