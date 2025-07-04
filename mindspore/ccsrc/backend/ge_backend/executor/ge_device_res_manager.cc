/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/executor/ge_device_res_manager.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "plugin/res_manager/ascend/symbol_interface/acl_rt_symbol.h"
#include "plugin/res_manager/ascend/symbol_interface/symbol_utils.h"
#include "plugin/res_manager/ascend/ascend_device_address/ascend_device_address.h"
#include "plugin/res_manager/ascend/hal_manager/ascend_hal_manager.h"
#include "runtime/device/res_manager/hal_res_manager.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
void GeDeviceResManager::Initialize() {
  if (initialized_) {
    return;
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  device::ResKey res_key{device::DeviceType::kAscend, device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);
  res_manager->Initialize();
  mem_manager_ = res_manager->mem_manager();
  MS_EXCEPTION_IF_NULL(mem_manager_);

  if (ms_context->get_param<bool>(MS_CTX_ENABLE_MEM_OFFLOAD)) {
    MS_LOG(WARNING) << "mem offload is not supported in ge.";
  }
  initialized_ = true;
}

void GeDeviceResManager::Destroy() {
  if (!initialized_) {
    return;
  }

  // memory Released in res_manager_->Destroy.
  mem_manager_ = nullptr;
  initialized_ = false;
}

bool GeDeviceResManager::AllocateMemory(device::DeviceAddress *const &address, uint32_t stream_id) const {
  MS_EXCEPTION_IF_NULL(address);

  if (address->GetDeviceType() != device::DeviceType::kAscend) {
    MS_LOG(EXCEPTION) << "The device address type is wrong: type name in address:"
                      << GetDeviceNameByType(static_cast<const device::DeviceType>(address->GetDeviceType()))
                      << ", type name in context: Ascend.";
  }

  if (address->GetPtr() != nullptr) {
    MS_LOG(ERROR) << "Memory leak detected!";
    return false;
  }

  BindDeviceToCurrentThread(false);

  if (stream_id == UINT32_MAX) {
    stream_id = address->stream_id();
  }
  MS_EXCEPTION_IF_NULL(mem_manager_);
  void *device_ptr = mem_manager_->MallocMemFromMemPool(address->GetSize(), address->from_persistent_mem(),
                                                        address->need_recycle(), stream_id);

  if (!device_ptr) {
    return false;
  }

  address->set_ptr(device_ptr);
  address->set_from_mem_pool(true);
  static std::string name = "Alloc memory";
  address->IncreaseNewRefCount(name);
  if (device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(BindDevicePtr, address, device_ptr);
  }
  return true;
}

void *GeDeviceResManager::AllocateMemory(size_t size, uint32_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(EXCEPTION) << "Bind context to current thread failed";
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->MallocMemFromMemPool(size, false, false, stream_id);
}

void *GeDeviceResManager::AllocateStaticMemory(size_t size, uint32_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(EXCEPTION) << "Bind context to current thread failed";
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->MallocMemFromMemPool(size, true, false, stream_id);
}

void *GeDeviceResManager::AllocateWorkSpaceMemory(size_t size) const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->ResetDynamicMemory();
  auto ptr = mem_manager_->MallocWorkSpaceMem(size);
  mem_manager_->ResetDynamicMemory();
  return ptr;
}

void GeDeviceResManager::FreeMemory(device::DeviceAddress *const &address) const {
  MS_EXCEPTION_IF_NULL(address);

  void *device_ptr = address->GetMutablePtr();
  if (device_ptr != nullptr) {
    if (!address->from_mem_pool()) {
      MS_LOG(DEBUG) << "device address:" << address << " ptr:" << device_ptr << " not from pool";
      return;
    }

    MS_LOG(DEBUG) << "Free memory from device address:" << address << " ptr:" << device_ptr;
    FreeMemory(device_ptr);
    address->set_ptr(nullptr);
  }
}

void GeDeviceResManager::FreeMemory(void *ptr) const {
  MS_EXCEPTION_IF_NULL(ptr);
  MS_EXCEPTION_IF_NULL(mem_manager_);
  mem_manager_->FreeMemFromMemPool(ptr);
}

size_t GeDeviceResManager::GetMaxUsedMemorySize() const {
  MS_EXCEPTION_IF_NULL(mem_manager_);
  return mem_manager_->GetMaxUsedMemorySize();
}

bool GeDeviceResManager::BindDeviceToCurrentThread(bool force_bind) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);

  if (force_bind) {
    device::ascend::AscendHalManager::GetInstance().SetContextForce(device_id);
  } else {
    device::ascend::AscendHalManager::GetInstance().SetContext(device_id);
  }
  return true;
}

void *GeDeviceResManager::GetStream() const { return device::ascend::AscendStreamMng::GetInstance().default_stream(); }

bool GeDeviceResManager::SyncStream(size_t stream_id) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return device::ascend::AscendStreamMng::GetInstance().SyncStream(stream_id);
}

bool GeDeviceResManager::SyncStream(void *stream) const {
  if (!BindDeviceToCurrentThread(false)) {
    MS_LOG(ERROR) << "Bind context to current thread failed";
    return false;
  }
  return device::ascend::AscendStreamMng::GetInstance().SyncStream(stream);
}

void *GeDeviceResManager::GetCopyDataStream() const {
  auto copy_data_stream = device::ascend::AscendStreamMng::GetInstance().GetCopyOutStream();
  if (copy_data_stream == nullptr) {
    size_t copy_stream_id;
    device::ascend::AscendStreamMng::GetInstance().CreateStream(&copy_stream_id);
    MS_LOG(INFO) << "Create ascend copy data stream, stream id: " << copy_stream_id;
    copy_data_stream = device::ascend::AscendStreamMng::GetInstance().GetStream(copy_stream_id);
    device::ascend::AscendStreamMng::GetInstance().SetCopyOutStream(copy_data_stream);
  }
  return copy_data_stream;
}

bool GeDeviceResManager::SyncCopyStream() const {
  auto copy_stream = GetCopyDataStream();
  MS_EXCEPTION_IF_NULL(copy_stream);
  return device::ascend::AscendStreamMng::GetInstance().SyncStream(copy_stream);
}

device::DeviceAddressPtr GeDeviceResManager::CreateDeviceAddress() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_address = std::make_shared<device::ascend::AscendDeviceAddress>();
  device_address->set_device_name(ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET));
  device_address->set_device_id(ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID));
  return device_address;
}

::ge::MemBlock *GeAllocator::Malloc(size_t size) {
  MS_EXCEPTION_IF_NULL(res_manager_);
  auto addr = res_manager_->AllocateMemory(size);
  MS_LOG(DEBUG) << "GE Allocator malloc addr: " << addr << " size: " << size;
  if (addr == nullptr) {
    MS_LOG(ERROR) << "GE Allocator malloc addr failed.";
    return nullptr;
  }
  auto mem_block = new ::ge::MemBlock(*this, addr, size);
  return mem_block;
}

void GeAllocator::Free(::ge::MemBlock *block) {
  if (res_manager_) {
    res_manager_->FreeMemory(block->GetAddr());
    MS_LOG(DEBUG) << "GE Allocator free addr: " << block->GetAddr();
  }
  delete block;
}
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
