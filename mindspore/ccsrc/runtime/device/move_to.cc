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
#include <string>
#include <memory>
#include <algorithm>
#include "runtime/device/move_to.h"
#include "common/device_type.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
namespace device {
namespace {
bool MoveToD2H(const tensor::TensorPtr &src_tensor, const DeviceAddressPtr &src_device_ptr,
               const tensor::TensorPtr &dst_tensor, bool blocking) {
  MS_EXCEPTION_IF_NULL(src_tensor);
  MS_EXCEPTION_IF_NULL(dst_tensor);
  if (src_device_ptr == nullptr) {
    MS_LOG(DEBUG) << "Origin tensor has no device address, just copy host value";
    size_t size = dst_tensor->Size();
    auto ret = memcpy_s(dst_tensor->data_c(), size, src_tensor->data_c(), size);
    return ret == EOK;
  }
  auto shape = src_tensor->shape();
  auto type_id = src_tensor->data_type();
  auto ret = true;
  std::string status;
  if (blocking) {
    status = "SyncDeviceToHost";
    ret = src_device_ptr->SyncDeviceToHost(shape, dst_tensor->Size(), type_id, dst_tensor->data_c());
  } else {
    status = "AsyncDeviceToHost";
    ret = src_device_ptr->AsyncDeviceToHost(dst_tensor->Size(), dst_tensor->data_c());
  }
  if (!ret) {
    MS_LOG(EXCEPTION) << status << " failed.";
  }
  return true;
}

void MoveToH2D(const tensor::TensorPtr &src_tensor, const DeviceAddressPtr &src_device_ptr,
               const DeviceAddressPtr &dst_device_ptr, bool blocking) {
  MS_EXCEPTION_IF_NULL(src_tensor);
  MS_EXCEPTION_IF_NULL(dst_device_ptr);
  auto shape = src_tensor->shape();
  auto type_id = src_tensor->data_type();
  auto src_size = src_tensor->Size();
  if (src_device_ptr != nullptr) {
    src_size = src_device_ptr->GetSize();
  }
  size_t size = std::min(src_size, dst_device_ptr->GetSize());
  auto src_data = src_device_ptr == nullptr ? src_tensor->data_c() : src_device_ptr->GetPtr();
  auto ret = true;
  std::string status;
  if (blocking) {
    ret = dst_device_ptr->SyncHostToDevice(shape, size, type_id, src_data);
    status = "SyncHostToDevice";
  } else {
    ret = dst_device_ptr->AsyncHostToDevice(size, src_data);
    status = "AsyncHostToDevice";
  }
  if (!ret) {
    MS_LOG(EXCEPTION) << status << " failed.";
  }
}
}  // namespace

void MoveTo(const tensor::TensorPtr &src_tensor, const tensor::TensorPtr &dst_tensor, const std::string &to,
            bool blocking, bool *return_self) {
  MS_EXCEPTION_IF_NULL(src_tensor);
  MS_EXCEPTION_IF_NULL(dst_tensor);
  MS_EXCEPTION_IF_NULL(return_self);

  const auto &device = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (to != "CPU" && to != device) {
    MS_LOG(EXCEPTION) << "The value of arg 'to' of method 'move_to' should be same with device target, bug got to:"
                      << to << ", device target: " << device;
  }

  auto src_addr = src_tensor->device_address();
  device::DeviceAddressPtr src_device_ptr = nullptr;
  if (src_addr != nullptr) {
    src_device_ptr = std::dynamic_pointer_cast<device::DeviceAddress>(src_addr);
    MS_EXCEPTION_IF_NULL(src_device_ptr);
    auto src_type = GetDeviceNameByType(src_device_ptr->GetDeviceType());
    if (to == src_type) {
      MS_LOG(DEBUG) << "The tensor is already on: " << to << ", no need move again";
      *return_self = true;
      return;
    }
  }
  // D2H copy, src_device_ptr: GPU/ASCEND; dst_device_ptr: CPU.
  if (to == "CPU") {
    if (src_device_ptr == nullptr) {
      MS_LOG(INFO) << "Src tensor device ptr is null, means tensor on: " << to << ", no need move again!";
      *return_self = true;
      return;
    }
    if (!MoveToD2H(src_tensor, src_device_ptr, dst_tensor, blocking)) {
      MS_LOG(EXCEPTION) << "Move tensor to " << to << "failed.";
    }
    dst_tensor->set_sync_status(kNeedSyncHostToDevice);
    return;
  }
  // H2D src_device_ptr: CPU; dst_device_ptr: GPU/ASCEND.
  auto dst_addr = std::dynamic_pointer_cast<device::DeviceAddress>(dst_tensor->device_address());
  if (dst_addr == nullptr) {
    auto size = src_device_ptr != nullptr ? src_device_ptr->GetSize() : src_tensor->Size();
    auto type_id = src_device_ptr != nullptr ? src_device_ptr->type_id() : src_tensor->data_type();
    auto host_shape = src_tensor->shape();
    auto device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto target_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({to, device_id});
    MS_EXCEPTION_IF_NULL(target_context);
    target_context->Initialize();
    auto stream_id = target_context->device_res_manager_->GetCurrentStreamId();
    if (target_context->device_res_manager_->GetStream(stream_id) == nullptr) {
      stream_id = kDefaultStreamIndex;
    }
    auto kernel_tensor = AnfAlgo::CreateKernelTensor(nullptr, size, kernel::GetFormatFromStrToEnum(kOpFormat_DEFAULT),
                                                     type_id, host_shape, to, device_id);
    dst_addr = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(dst_addr);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, "PyNative", memory::mem_pool::MemType::kPyNativeOutput,
                                                   dst_addr->GetSize(), dst_addr.get());
    if (!target_context->device_res_manager_->AllocateMemory(dst_addr.get(), stream_id)) {
      MS_LOG(EXCEPTION) << "Allocate memory failed, maybe device memory(device id:" << device_id
                        << ") isn't enough. Allocate size: " << size;
    }
    dst_tensor->set_device_address(dst_addr);
  }
  MoveToH2D(src_tensor, src_device_ptr, dst_addr, blocking);
  dst_tensor->set_sync_status(kNeedSyncDeviceToHost);
}
}  // namespace device
}  // namespace mindspore
