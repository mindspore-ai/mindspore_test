/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#include "plugin/res_manager/gpu/device/gpu_device_address.h"
#include <vector>
#include <memory>
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "ir/tensor.h"
#include "plugin/res_manager/gpu/device/gpu_device_manager.h"
#include "plugin/res_manager/gpu/device/gpu_memory_allocator.h"
#include "plugin/res_manager/gpu/device/gpu_hash_table_util.h"
#include "plugin/device/gpu/hal/device/gpu_common.h"
#include "plugin/res_manager/gpu/device/gpu_event.h"
#include "runtime/device/res_manager/hal_res_manager.h"
#include "plugin/res_manager/gpu/device/gpu_device_synchronizer.h"

namespace mindspore {
namespace device {
namespace gpu {
DeviceSynchronizerPtr GPUDeviceAddress::NewDeviceSynchronizer() { return std::make_shared<GPUDeviceSynchronizer>(); }

void GPUDeviceAddress::SetDevicePtrDeleter() {
  if (address_common_ == nullptr || address_common_->pointer_ref_count_ == nullptr) {
    return;
  }
  address_common_->pointer_ref_count_->set_deleter([](void *ptr, bool from_mem_pool) {
    if (ptr != nullptr && from_mem_pool) {
      GPUMemoryAllocator::GetInstance().FreeTensorMem(ptr);
    }
  });
}

bool GPUDeviceAddress::SyncDeviceToHost(size_t size, void *host_ptr) const {
  // The input or output may be empty.
  if ((size == 0) || (GetSize() == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << GetSize();
    return true;
  }
  if (size > GetSize()) {
    MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << GetSize();
    return true;
  }

  MS_EXCEPTION_IF_NULL(host_ptr);
  auto ret = GPUDeviceManager::GetInstance().SyncAllStreams();
  if (!ret) {
    MS_LOG(ERROR) << "SyncStream failed";
    return ret;
  }
  if (size != GetSize()) {
    // nccl kernel input and output device address is aligned, may lead to host size is not equal to device size
    MS_LOG(INFO) << "Sync memory size is inconsistent, host size: " << size << ", device size " << GetSize();
  }
  MoveToDevice(false);
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  MS_EXCEPTION_IF_NULL(GetDevicePtr());
  return GPUDeviceManager::GetInstance().CopyDeviceMemToHost(host_ptr, GetDevicePtr(), size);
}

bool GPUDeviceAddress::SyncHostToDevice(size_t size, const void *host_ptr) const {
  // The input or output may be empty.
  if ((size == 0) || (GetSize() == 0)) {
    MS_LOG(INFO) << "No need sync, host size: " << size << ", device size: " << GetSize();
    return true;
  }
  if (size > GetSize()) {
    MS_LOG(WARNING) << "Please check whether need sync data, host size: " << size << ", device size: " << GetSize();
    return true;
  }
  MS_EXCEPTION_IF_NULL(host_ptr);
  if (size != GetSize()) {
    // nccl kernel input and output device address is aligned, may lead to host size is not equal to device size
    MS_LOG(INFO) << "Sync memory size is inconsistent, host size: " << size << ", device size " << GetSize();
  }

  // Bind device by device name and device id on the current thread.
  if (!device_name().empty()) {
    auto ms_context = MsContext::GetInstance();
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto gpu_res_manager = HalResManager::GetInstance().GetOrCreateResManager({DeviceType::kGPU, device_id});
    MS_EXCEPTION_IF_NULL(gpu_res_manager);
    if (!gpu_res_manager->BindDeviceToCurrentThread(false)) {
      MS_LOG(EXCEPTION) << "BindDeviceToCurrentThread failed.";
    }
  }

  MoveToDevice(false);
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  MS_EXCEPTION_IF_NULL(GetDevicePtr());
  auto stream = GPUDeviceManager::GetInstance().GetStream(this->stream_id());
  MS_EXCEPTION_IF_NULL(stream);
  if (!GPUDeviceManager::GetInstance().CopyHostMemToDeviceAsync(GetDevicePtr(), host_ptr, size, stream)) {
    MS_LOG(ERROR) << "CopyHostMemToDeviceAsync failed";
    return false;
  }
  return GPUDeviceManager::GetInstance().SyncStream(stream);
}

bool GPUDeviceAddress::SyncDeviceToHost(const ShapeVector &, size_t size, TypeId, void *host_ptr,
                                        bool sync_on_demand) const {
  return SyncDeviceToHost(size, host_ptr);
}

namespace {
bool SyncUserDataToDevice(const UserDataPtr &user_data, const void *host_ptr, size_t size) {
  MS_EXCEPTION_IF_NULL(user_data);
  MS_EXCEPTION_IF_NULL(host_ptr);
  const auto &user_data_type = user_data->get<UserDataType>(kUserDataType);
  MS_EXCEPTION_IF_NULL(user_data_type);

  if (*user_data_type == UserDataType::kUserTypeHashTable) {
#if CUDA_VERSION > 11000 && defined(__linux__)
    auto key_type = user_data->get<TypeId>(kHashTableKeyType);
    auto value_type = user_data->get<TypeId>(kHashTableValueType);
    MS_EXCEPTION_IF_NULL(key_type);
    MS_EXCEPTION_IF_NULL(value_type);
    const auto &iter = hashtable_func_list.find({*key_type, *value_type});
    if (iter != hashtable_func_list.end()) {
      return std::get<kSyncFuncIndex>(iter->second)(user_data, host_ptr, size);
    } else {
      MS_LOG(EXCEPTION) << "Unsupported hash table type:" << *key_type << " and:" << *value_type;
    }
#else
    MS_LOG(EXCEPTION) << "Invalid platform or cuda version for gpu hash table.";
#endif
  }
  return true;
}
}  // namespace

bool GPUDeviceAddress::SyncHostToDevice(const ShapeVector &, size_t size, TypeId, const void *host_ptr,
                                        const std::string &format) const {
  if (user_data() != nullptr && user_data()->has(kUserDataType)) {
    return SyncUserDataToDevice(user_data(), host_ptr, size);
  }

  MoveToDevice(false);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool execution_mode = ms_context->get_param<int>(MS_CTX_EXECUTION_MODE);
  if (execution_mode != kPynativeMode) {
    return SyncHostToDevice(size, host_ptr);
  }

  // PyNative mode need copy async to improve performance.
  MS_EXCEPTION_IF_NULL(host_ptr);
  bool need_sync = (size != 0) && (GetSize() != 0) && (size <= GetSize());
  if (!need_sync) {
    return true;
  }
  auto stream = GPUDeviceManager::GetInstance().GetStream(this->stream_id());
  MS_EXCEPTION_IF_NULL(stream);
  return GPUDeviceManager::GetInstance().CopyHostMemToDeviceAsync(GetDevicePtr(), host_ptr, size, stream);
}

bool GPUDeviceAddress::SyncDeviceToDevice(const DeviceSync *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  auto src_gpu_device = dynamic_cast<const GPUDeviceAddress *>(src_device_addr);
  MS_EXCEPTION_IF_NULL(src_gpu_device);
  MS_LOG(DEBUG) << "Sync gpu device address from:" << src_device_addr << " to:" << this;
  src_gpu_device->MoveToDevice(false);

  return SyncDeviceToDevice(src_gpu_device->host_shape(), src_gpu_device->GetSize(), src_gpu_device->type_id(),
                            src_gpu_device->GetPtr(), src_gpu_device->format());
}

bool GPUDeviceAddress::SyncDeviceToDevice(const ShapeVector &shape, size_t size, TypeId type, const void *src_ptr,
                                          const std::string &format) const {
  bool ret = AsyncDeviceToDevice(shape, size, type, src_ptr, format);
  if (!ret) {
    return ret;
  }
  auto &stream = GPUDeviceManager::GetInstance().default_stream();
  return GPUDeviceManager::GetInstance().SyncStream(stream);
}

bool GPUDeviceAddress::AsyncDeviceToDevice(const DeviceAddress *src_device_addr) const {
  MS_EXCEPTION_IF_NULL(src_device_addr);
  MS_LOG(DEBUG) << "Async gpu device address from:" << src_device_addr << " to:" << this;
  return AsyncDeviceToDevice(ShapeVector(), src_device_addr->GetSize(), src_device_addr->type_id(),
                             src_device_addr->GetPtr(), src_device_addr->format());
}

bool GPUDeviceAddress::AsyncDeviceToDevice(const ShapeVector &, size_t size, TypeId type, const void *src_ptr,
                                           const std::string &format) const {
  MS_LOG(DEBUG) << "AsyncDeviceToDevice, dst(address:" << GetDevicePtr() << " format:" << DeviceAddress::format()
                << ", type_id:" << TypeIdLabel(type_id()) << ", size:" << GetSize() << "), src(address:" << src_ptr
                << "format:" << format << ", type_id:" << TypeIdLabel(type) << ", size:" << size << ")";
  if (GetDevicePtr() == src_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need memcpy data.";
    return true;
  }
  if (type_id() > kMonadTypeBegin && type_id() < kMonadTypeEnd) {
    return true;
  }
  // The input or output may be empty.
  if ((size == 0) || (GetSize() == 0)) {
    MS_LOG(INFO) << "No need sync, src device size: " << size << ", dst device size: " << GetSize();
    return true;
  }
  if (GetSize() < size) {
    MS_LOG(ERROR) << "Src size is greater than det size, src size is: " << size << ", dst size is: " << GetSize();
    return false;
  }
  if (DeviceAddress::format() != format || type_id() != type) {
    MS_LOG(ERROR) << "Format or type is different, src(format:" << format << ", type_id:" << TypeIdLabel(type)
                  << "), dst(format:" << DeviceAddress::format() << "), type_id:" << TypeIdLabel(type_id());
    return false;
  }

  MoveToDevice(false);
  MS_EXCEPTION_IF_NULL(src_ptr);
  MS_EXCEPTION_IF_NULL(GetDevicePtr());
  auto &stream = GPUDeviceManager::GetInstance().default_stream();
  MS_EXCEPTION_IF_NULL(stream);
  if (!GPUDeviceManager::GetInstance().CopyDeviceMemToDeviceAsync(GetDevicePtr(), src_ptr, size, stream)) {
    MS_LOG(ERROR) << "CopyDeviceMemToDeviceAsync failed";
    return false;
  }
  return true;
}

bool GPUDeviceAddress::AsyncHostToDevice(size_t size, const void *host_ptr) const {
  MS_ERROR_IF_NULL(host_ptr);
  if (GetDevicePtr() == host_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need copy data.";
    return true;
  }
  auto device_res_manager = GetHalRes();
  MS_ERROR_IF_NULL(device_res_manager);
  auto stream_id = device_res_manager->GetCurrentStreamId();
  auto stream = device_res_manager->GetStream(stream_id);
  if (stream == nullptr) {
    stream = device_res_manager->GetStream(kDefaultStreamIndex);
    stream_id = kDefaultStreamIndex;
  }
  MS_ERROR_IF_NULL(stream);
  if (GetDevicePtr() == nullptr) {
    auto ptr = device_res_manager->AllocateMemory(size, stream_id);
    MS_EXCEPTION_IF_NULL(ptr);
    SetDevicePtr(ptr);
  }
  CHECK_RET_WITH_RETURN_ERROR(CudaDriver::CopyHostMemToDeviceAsync(GetDevicePtr(), host_ptr, size, stream),
                              "CopyHostMemToDeviceAsync failed");

  return true;
}

bool GPUDeviceAddress::AsyncDeviceToHost(size_t size, void *host_ptr) const {
  MS_ERROR_IF_NULL(host_ptr);
  MS_ERROR_IF_NULL(GetDevicePtr());
  if (GetDevicePtr() == host_ptr) {
    MS_LOG(INFO) << "Dst addr is same with src addr, no need copy data.";
    return true;
  }
  auto device_res_manager = GetHalRes();
  MS_ERROR_IF_NULL(device_res_manager);
  auto stream_id = device_res_manager->GetCurrentStreamId();
  auto stream = device_res_manager->GetStream(stream_id);
  if (stream == nullptr) {
    stream = device_res_manager->GetStream(kDefaultStreamIndex);
  }
  MS_ERROR_IF_NULL(stream);
  CHECK_RET_WITH_RETURN_ERROR(CudaDriver::CopyDeviceMemToHostAsync(host_ptr, GetDevicePtr(), size, stream),
                              "CopyHostMemToDeviceAsync failed");
  return true;
}

bool GPUDeviceAddress::AsyncHostToDevice(size_t size, TypeId type, const tensor::TensorDataPtr &tensor_data,
                                         const std::string &format) const {
  return AsyncHostToDevice(size, tensor_data->data());
}

bool GPUDeviceAddress::AsyncHostToDevice(const ShapeVector &, size_t size, TypeId, const void *host_ptr,
                                         size_t stream_id) const {
  MS_ERROR_IF_NULL(host_ptr);
  MoveToDevice(false);
  MS_ERROR_IF_NULL(GetDevicePtr());
  const auto stream = GPUDeviceManager::GetInstance().GetStream(stream_id);
  MS_ERROR_IF_NULL(stream);

  CHECK_RET_WITH_RETURN_ERROR(CudaDriver::CopyHostMemToDeviceAsync(GetDevicePtr(), host_ptr, size, stream),
                              "CopyHostMemToDeviceAsync failed");
  return true;
}

bool GPUDeviceAddress::AsyncDeviceToHost(const ShapeVector &, size_t size, TypeId, void *host_ptr,
                                         size_t stream_id) const {
  MS_ERROR_IF_NULL(host_ptr);
  MoveToDevice(false);
  MS_ERROR_IF_NULL(GetDevicePtr());
  const auto stream = GPUDeviceManager::GetInstance().GetStream(stream_id);
  MS_ERROR_IF_NULL(stream);

  CHECK_RET_WITH_RETURN_ERROR(CudaDriver::CopyDeviceMemToHostAsync(host_ptr, GetDevicePtr(), size, stream),
                              "CopyHostMemToDeviceAsync failed");
  return true;
}

void GPUDeviceAddress::ClearDeviceMemory() {
  std::lock_guard<std::recursive_mutex> lock(ptr_mutex_);
  if (GetDevicePtr() != nullptr && from_mem_pool()) {
    GPUMemoryAllocator::GetInstance().FreeTensorMem(GetDevicePtr());
    SetDevicePtr(nullptr);
  }
}

void GPUDeviceAddress::ClearUserData() {
  if (user_data() == nullptr || (!user_data()->has(kUserDataType))) {
    return;
  }

  auto user_data_type = user_data()->get<UserDataType>(kUserDataType);
  MS_EXCEPTION_IF_NULL(user_data_type);
  if (*user_data_type == UserDataType::kUserTypeHashTable) {
#if CUDA_VERSION > 11000 && defined(__linux__)
    auto key_type = user_data()->get<TypeId>(kHashTableKeyType);
    auto value_type = user_data()->get<TypeId>(kHashTableValueType);
    MS_EXCEPTION_IF_NULL(key_type);
    MS_EXCEPTION_IF_NULL(value_type);
    const auto &iter = hashtable_func_list.find({*key_type, *value_type});
    if (iter != hashtable_func_list.end()) {
      return std::get<kClearFuncIndex>(iter->second)(user_data());
    } else {
      MS_LOG(EXCEPTION) << "Unsupported hash table type:" << *key_type << " and:" << *value_type;
    }
#else
    MS_LOG(EXCEPTION) << "Invalid platform or cuda version for gpu hash table.";
#endif
  }
}

GPUDeviceAddress::~GPUDeviceAddress() { LoadableDeviceAddress::ReleaseResource(); }

bool GPUDeviceAddress::CopyBetweenHostDevice(void *dst, const void *src, size_t size, bool async, size_t stream_id,
                                             bool host_to_device) const {
  MS_ERROR_IF_NULL(dst);
  MS_ERROR_IF_NULL(src);
  const auto stream = GPUDeviceManager::GetInstance().GetStream(stream_id);
  MS_ERROR_IF_NULL(stream);
  if (host_to_device) {
    CHECK_RET_WITH_RETURN_ERROR(CudaDriver::CopyHostMemToDeviceAsync(dst, src, size, stream),
                                "CopyHostMemToDeviceAsync failed");
  } else {
    CHECK_RET_WITH_RETURN_ERROR(CudaDriver::CopyDeviceMemToHostAsync(dst, src, size, stream),
                                "CopyDeviceMemToHostAsync failed");
  }
  if (async) {
    auto record_event = std::make_shared<GpuEvent>();
    record_event->set_record_stream(stream);
    record_event->RecordEvent();
    if (loadable_mem_ == nullptr) {
      loadable_mem_ = std::make_unique<LoadableMember>();
    }
    loadable_mem_->swap_event_.device_event_ = record_event;
  } else {
    GPUDeviceManager::GetInstance().SyncStream(stream);
  }
  return true;
}

bool GPUDeviceAddress::CopyDeviceToHost(void *dst, const void *src, size_t size, bool async, size_t stream_id) const {
  return CopyBetweenHostDevice(dst, src, size, async, stream_id, false);
}

bool GPUDeviceAddress::CopyHostToDevice(void *dst, const void *src, size_t size, bool async, size_t stream_id) const {
  return CopyBetweenHostDevice(dst, src, size, async, stream_id, true);
}

bool GPUDeviceAddress::CopyHostToDevice(void *dst, const void *src, const size_t &size) const {
  return GPUDeviceManager::GetInstance().CopyHostMemToDevice(dst, src, size);
}

bool GPUDeviceAddress::CopyDeviceToHost(void *dst, const void *src, const size_t &size) const {
  return GPUDeviceManager::GetInstance().CopyDeviceMemToHost(dst, const_cast<void *>(src), size);
}

DeviceAddressPtr GPUDeviceAddress::CloneDeviceAddress() {
  auto clone_device_address = std::make_shared<GPUDeviceAddress>();
  DeviceAddress::CloneDeviceAddress(clone_device_address);
  return clone_device_address;
}

/*
 * Feature group: Dump, Online debugger.
 * Target device group: GPU.
 * Runtime category: Old runtime, MindRT.
 * Description: Load tensor to host and create tensor_data object for the loaded tensor.
 */
mindspore::tensor::TensorPtr GPUDeviceAddress::LoadMemToHost(const std::string &tensor_name,
                                                             const ShapeVector &host_shape, TypeId host_type, bool,
                                                             bool) const {
  mindspore::tensor::TensorPtr out_tensor = std::make_shared<tensor::Tensor>(host_type, host_shape);
  size_t host_size = out_tensor->data().nbytes();
  if (host_size == 0) {
    MS_LOG(INFO) << "Host size is 0 for tensor: " << tensor_name << ", no need to load.";
    return std::make_shared<mindspore::tensor::Tensor>();
  }
  auto ret_rt_memcpy = SyncDeviceToHost(host_shape, host_size, host_type, out_tensor->data_c());
  if (!ret_rt_memcpy) {
    MS_LOG(ERROR) << "Copy device mem to host failed";
    return nullptr;
  }
  return out_tensor;
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
