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

#include "plugin/device/ascend/llm_boost/atb/buffer_device.h"
#include "runtime/hardware/device_context_manager.h"
#include "acl/acl.h"

namespace mindspore {
namespace kernel {
constexpr int KB_1 = 1024;
constexpr int MB_1 = 1024 * 1024;
constexpr int GB_1 = 1024 * 1024 * 1024;

BufferDevice::BufferDevice(uint64_t bufferSize) : bufferSize_(bufferSize) {
  MS_LOG(INFO) << "BufferDevice::BufferDevice called, bufferSize:" << bufferSize;
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET), ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_context_);

  bufferSize_ = bufferSize;
  if (bufferSize_ > 0) {
    MS_LOG(INFO) << "BufferDevice::GetBuffer bufferSize:" << bufferSize_;
    buffer_ = device_context_->device_res_manager_->AllocateMemory(bufferSize);
  }
}

BufferDevice::~BufferDevice() {}

void *BufferDevice::GetBuffer(uint64_t bufferSize) {
  if (bufferSize <= bufferSize_) {
    MS_LOG(INFO) << "BufferDevice::GetBuffer bufferSize:" << bufferSize << "<= bufferSize_:" << bufferSize_
                 << ", not new device mem.";
    return buffer_;
  }

  if (aclrtSynchronizeDevice() != 0) {
    return nullptr;
  }

  bufferSize_ = bufferSize;
  MS_LOG(INFO) << "BufferDevice::GetBuffer new bufferSize:" << bufferSize;
  buffer_ = device_context_->device_res_manager_->AllocateMemory(bufferSize);
  return buffer_;
}
}  // namespace kernel
}  // namespace mindspore
