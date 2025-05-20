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
#include "frontend/ir/storage_base.h"
#include <utility>
#include <string>
#include "runtime/device/res_manager/utils/utils.h"
#include "runtime/device/res_manager/hal_res_manager.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/pipeline/pipeline.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "mindspore/ccsrc/include/backend/mem_reuse/mem_tracker.h"

namespace mindspore {
namespace {
device::DeviceAddressPtr CreateTempDeviceAddress(const device::DeviceAddressPtr &device_address,
                                                 const DeviceContext *device_context) {
  ShapeVector shape = {static_cast<int64_t>(device_address->size())};
  auto new_device_address = device_context->device_res_manager_->CreateDeviceAddress(
    device_address->GetMutablePtr(), device_address->size(), shape, device_address->address_common()->format_,
    device_address->type_id(), device_address->device_name(), device_address->device_id(), device_address->stream_id(),
    device_address->user_data());
  new_device_address->set_from_mem_pool(false);
  return new_device_address;
}
};  // namespace

StorageBase::~StorageBase() { device_data_ = nullptr; }

uintptr_t StorageBase::DataPtr() const {
  if (device_data_ != nullptr) {
    auto *data_ptr = device_data_->GetMutablePtr();
    return reinterpret_cast<uintptr_t>(data_ptr);
  }

  return reinterpret_cast<uintptr_t>(nullptr);
}

void StorageBase::InplaceReSize(int64_t size) {
  runtime::Pipeline::Get().WaitForward();
  if (device_data_ != nullptr) {
    if (size == 0) {
      device_data_->ClearDeviceMemory();
      device_data_->SetSize(0);
      return;
    }

    device::ResKey res_key{device::GetDeviceTypeByName(device_data_->device_name()), device_data_->device_id()};
    auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);
    void *device_ptr = nullptr;
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "ResizeStorage", "ResizeStorage", "");
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "ResizeStorage", memory::mem_pool::MemType::kOther, size,
                                                   device_data_.get());
    device_ptr = res_manager->AllocateMemory(size, device_data_->stream_id());
    if (!device_ptr) {
      return;
    }
    device_data_->set_ptr(device_ptr);
    device_data_->set_from_mem_pool(true);
    static std::string name = "Alloc memory";
    device_data_->IncreaseNewRefCount(name);
    device_data_->SetSize(size);
    return;
  }

  MS_LOG(EXCEPTION) << "The current Storage does not yet support resize for CPU";
}

int64_t StorageBase::NBytes() const {
  if (device_data_ != nullptr) {
    return device_data_->size();
  }

  return 0;
}

void StorageBase::InplaceCopy(const StorageBasePtr &src, bool non_blocking) {
  if (non_blocking == true) {
    MS_LOG(WARNING) << "The current Storage does not yet support non-blocking copy.";
  }
  runtime::Pipeline::Get().WaitAll();
  if (device_data_ != nullptr && src->device_data_ != nullptr) {
    auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_data_->device_name(), device_data_->device_id()});
    MS_EXCEPTION_IF_NULL(device_context);
    device_context->Initialize();
    auto dst_address = device_data_;
    auto src_address = src->device_data_;
    if (device_data_->address_common()->shape_vector_ != src->device_data_->address_common()->shape_vector_) {
      dst_address = CreateTempDeviceAddress(dst_address, device_context);
      src_address = CreateTempDeviceAddress(src_address, device_context);
    }
    if (!device_context->GetKernelExecutor(false)->ExecuteKernelTask(
          runtime::KernelTaskType::kCOPY_TASK, {dst_address, src_address}, {}, device_data_->stream_id())) {
      MS_LOG(EXCEPTION) << "ExecuteKernelTask failed, task_type: " << runtime::KernelTaskType::kCOPY_TASK;
    }
    runtime::Pipeline::Get().WaitForward();
    auto &controller = device::HalResManager::GetInstance().GetMultiStreamController(device_context->DeviceName());
    controller->Refresh();
    (void)controller->SyncStream(device_data_->stream_id());
    return;
  }

  MS_LOG(EXCEPTION) << "The current Storage does not yet support CPU copy";
  return;
}

std::string StorageBase::device() const {
  if (device_data_ != nullptr) {
    return device_data_->device_name();
  }

  MS_LOG(EXCEPTION) << "The current Storage does not yet support CPU";
  return "";
}
}  // namespace mindspore
