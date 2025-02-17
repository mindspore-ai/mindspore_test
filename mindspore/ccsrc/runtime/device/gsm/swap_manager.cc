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

#include "runtime/device/gsm/swap_manager.h"

#include <functional>
#include <string>
#include <utility>

#include "include/common/utils/offload_context.h"
#include "include/common/debug/common.h"
#include "utils/file_utils.h"
#include "utils/temp_file_manager.h"

namespace mindspore {
namespace device {
constexpr char kSwapFileSuffix[] = ".data";
constexpr char kLinuxAioLibName[] = "libaio_plugin.so";
constexpr char kLinuxAioInstanceFuncName[] = "get_aio_instance";
constexpr size_t kSizeLevelNum = 8;
constexpr size_t kSwapMemAlignSize = 512;

SwapManager::SwapManager(size_t stream_id, mindspore::device::DynamicMemPool *device_memory_pool,
                         PinMemPool *pin_mem_pool)
    : stream_id_(stream_id),
      device_memory_pool_(device_memory_pool),
      pin_mem_pool_(pin_mem_pool),
      size_level_num_(kSizeLevelNum) {
  const auto &offload_context = OffloadContext::GetInstance();
  io_handle_ = std::make_shared<IOHandle>();
  if (offload_context != nullptr) {
    if (offload_context->enable_aio()) {
      io_handle_->LoadAio(kLinuxAioLibName, kLinuxAioInstanceFuncName);
    }
    max_file_size_ = offload_context->offload_disk_size();
  }
  MS_EXCEPTION_IF_NULL(offload_context);
  (void)FileUtils::CreateNotExistDirs(offload_context->offload_path(), true);
}

template <class Input, class Output>
bool SwapManager::TryAllocate(std::queue<const DeviceAddress *> queue, const Input &input, uint32_t stream_id,
                              Output (SwapManager::*allocate_func)(const Input &, uint32_t),
                              const std::function<bool(Output)> &success, Output *output) {
  MS_EXCEPTION_IF_NULL(allocate_func);
  MS_EXCEPTION_IF_NULL(output);
  (*output) = (this->*allocate_func)(input, stream_id);
  if (success(*output)) {
    return true;
  }
  // Wait swapping tensors.
  while (!queue.empty()) {
    const auto &front = queue.front();
    MS_EXCEPTION_IF_NULL(front);
    if (front->Wait()) {
      (*output) = (this->*allocate_func)(input, stream_id);
      if (success(*output)) {
        return true;
      }
    }
    queue.pop();
  }
  return false;
}

void *SwapManager::AllocDeviceMemorySimply(const size_t &size, uint32_t stream_id) {
  MS_EXCEPTION_IF_NULL(device_memory_pool_);
  return device_memory_pool_->AllocTensorMem(size + kSwapMemAlignSize, false, false, stream_id);
}

void *SwapManager::AllocDeviceMemory(size_t size, uint32_t stream_id) {
  void *ret = nullptr;
  void *(SwapManager::*allocate_func)(const size_t &, uint32_t) = &SwapManager::AllocDeviceMemorySimply;
  std::function<bool(void *)> success = [](void *ptr) { return ptr != nullptr; };
  std::lock_guard<std::mutex> lock(swapping_tensors_device_mutex_);
  if (!TryAllocate(swapping_tensors_device_, size, stream_id, allocate_func, success, &ret)) {
    MS_LOG(WARNING) << "Allocate device memory failed, size: " << size;
  }
  return ret;
}

std::vector<void *> SwapManager::AllocDeviceContinuousMemSimply(const std::vector<size_t> &size_list,
                                                                uint32_t stream_id) {
  MS_EXCEPTION_IF_NULL(device_memory_pool_);
  return device_memory_pool_->AllocContinuousTensorMem(size_list, stream_id);
}

std::vector<void *> SwapManager::AllocDeviceContinuousMem(const std::vector<size_t> &size_list, uint32_t stream_id) {
  std::vector<void *> ret;
  std::vector<void *> (SwapManager::*allocate_func)(const std::vector<size_t> &, uint32_t) =
    &SwapManager::AllocDeviceContinuousMemSimply;
  std::function<bool(std::vector<void *>)> success = [](const std::vector<void *> &ptrs) { return !ptrs.empty(); };
  std::lock_guard<std::mutex> lock(swapping_tensors_device_mutex_);
  if (!TryAllocate(swapping_tensors_device_, size_list, stream_id, allocate_func, success, &ret)) {
    MS_LOG(WARNING) << "Allocate continuous device mem failed, size list: " << size_list;
  }
  return ret;
}

void SwapManager::FreeDeviceMemory(void *ptr) {
  MS_EXCEPTION_IF_NULL(device_memory_pool_);
  device_memory_pool_->FreeTensorMem(ptr);
}

void *SwapManager::AllocHostMemorySimply(const size_t &size, uint32_t /*stream_id*/) {
  MS_EXCEPTION_IF_NULL(pin_mem_pool_);
  return pin_mem_pool_->AllocPinMem(size);
}

void *SwapManager::AllocHostMemory(size_t size) {
  void *ret = nullptr;
  void *(SwapManager::*allocate_func)(const size_t &, uint32_t) = &SwapManager::AllocHostMemorySimply;
  std::function<bool(void *)> success = [](void *ptr) { return ptr != nullptr; };
  std::lock_guard<std::mutex> lock(swapping_tensors_host_mutex_);
  if (!TryAllocate(swapping_tensors_host_, size, kDefaultStreamIndex, allocate_func, success, &ret)) {
    MS_LOG(WARNING) << "Allocate host memory failed, size: " << size;
  }
  return ret;
}

void SwapManager::FreeHostMemory(void *ptr) {
  MS_EXCEPTION_IF_NULL(pin_mem_pool_);
  pin_mem_pool_->FreeTensorMem(ptr);
}

bool SwapManager::CreateFile(const std::string &file_name, size_t file_size) {
  MS_EXCEPTION_IF_NULL(io_handle_);
  bool (SwapManager::*allocate_func)(const size_t &size, uint32_t) = &SwapManager::EnoughFileSpace;
  std::function<bool(bool)> success = [](bool ret) { return ret; };
  {
    std::lock_guard<std::mutex> lock(swapping_tensors_file_mutex_);
    bool enough = false;
    if (!TryAllocate(swapping_tensors_file_, file_size, kDefaultStreamIndex, allocate_func, success, &enough)) {
      MS_LOG(WARNING) << "There is no enough disk space for creating file, size: " << file_size;
      return false;
    }
  }
  current_used_file_size_ += file_size;
  file_size_[file_name] = file_size;
  TempFileManager::GetInstance().Register(file_name);
  return io_handle_->CreateSwapFile(file_name);
}

bool SwapManager::DeleteFile(const std::string &file_name) {
  MS_EXCEPTION_IF_NULL(io_handle_);
  const auto &iter = file_size_.find(file_name);
  if (iter == file_size_.end()) {
    MS_LOG(WARNING) << "Can not file size for file[" << file_name << "]";
  } else {
    current_used_file_size_ -= iter->second;
    iter->second = 0;
  }
  TempFileManager::GetInstance().UnRegister(file_name);
  return io_handle_->DeleteSwapFile(file_name);
}

bool SwapManager::FileToHostMemory(void *host_memory, const std::string &file_name, size_t byte_num, bool async,
                                   AsyncIOToken *sync_key) {
  MS_EXCEPTION_IF_NULL(io_handle_);
  if (async) {
    return io_handle_->ReadAsync(file_name, host_memory, byte_num, sync_key);
  } else {
    return io_handle_->Read(file_name, host_memory, byte_num);
  }
}

bool SwapManager::EnoughFileSpace(const size_t &size, uint32_t /*stream_id*/) {
  return current_used_file_size_ + size <= max_file_size_;
}

bool SwapManager::HostMemoryToFile(const std::string &file_name, const void *data, size_t byte_num, bool async,
                                   AsyncIOToken *sync_key) {
  MS_EXCEPTION_IF_NULL(io_handle_);
  if (async) {
    return io_handle_->WriteAsync(file_name, data, byte_num, sync_key);
  } else {
    return io_handle_->Write(file_name, data, byte_num);
  }
}

bool SwapManager::WaitAsyncIO(mindspore::device::AsyncIOToken sync_token) {
  MS_EXCEPTION_IF_NULL(io_handle_);
  return io_handle_->Wait(sync_token);
}

std::string SwapManager::GetSwapFileName(uint32_t device_id) const {
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
  return file_dir + std::to_string(device_id) + "_" + std::to_string(swap_file_index++) + "_" +
         std::to_string(Common::GetTimeStamp()) + kSwapFileSuffix;
}

void SwapManager::AddSwappingTensor(const mindspore::device::DeviceAddress *device_address) {
  if (device_address == nullptr) {
    return;
  }
  if (device_address->status() == DeviceAddressStatus::kInFileToHost) {
    std::lock_guard<std::mutex> lock(swapping_tensors_file_mutex_);
    (void)swapping_tensors_file_.push(device_address);
  } else if (device_address->status() == DeviceAddressStatus::kInDeviceToHost) {
    std::lock_guard<std::mutex> lock(swapping_tensors_device_mutex_);
    (void)swapping_tensors_device_.push(device_address);
  } else {
    std::lock_guard<std::mutex> lock(swapping_tensors_host_mutex_);
    (void)swapping_tensors_host_.push(device_address);
  }
}
}  // namespace device
}  // namespace mindspore
