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

#include "ir/device_sync.h"

namespace mindspore {
CopyFunc g_sync_copy_func[static_cast<int>(device::DeviceType::kDeviceEnd)];
CopyFunc g_async_copy_func[static_cast<int>(device::DeviceType::kDeviceEnd)];

MS_CORE_API void SetCopyFunc(device::DeviceType device_type, CopyFunc &&sync_func, CopyFunc &&async_func) {
  MS_LOG(WARNING) << "Resigter copy function for device type:" << device_type;
  g_sync_copy_func[static_cast<int>(device_type)] = sync_func;
  g_async_copy_func[static_cast<int>(device_type)] = async_func;
}

bool SyncCopy(const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync, size_t stream_id) {
  MS_EXCEPTION_IF_NULL(dst_device_sync);
  MS_EXCEPTION_IF_NULL(src_device_sync);
  if (dst_device_sync->GetDeviceType() == device::DeviceType::kUnknown ||
      src_device_sync->GetDeviceType() == device::DeviceType::kUnknown) {
    MS_LOG(EXCEPTION) << "Invalid device type for device sync:" << dst_device_sync
                      << " type:" << dst_device_sync->GetDeviceType() << " or device sync:" << src_device_sync
                      << " type:" << src_device_sync->GetDeviceType() << " stream id:" << stream_id;
  }
  if (dst_device_sync->GetDeviceType() == device::DeviceType::kCPU &&
      src_device_sync->GetDeviceType() == device::DeviceType::kCPU) {
    MS_EXCEPTION_IF_NULL(g_sync_copy_func[static_cast<int>(device::DeviceType::kCPU)]);
    return g_sync_copy_func[static_cast<int>(device::DeviceType::kCPU)](dst_device_sync, src_device_sync, stream_id);
  }
  if (dst_device_sync->GetDeviceType() == device::DeviceType::kAscend ||
      src_device_sync->GetDeviceType() == device::DeviceType::kAscend) {
    MS_EXCEPTION_IF_NULL(g_sync_copy_func[static_cast<int>(device::DeviceType::kAscend)]);
    return g_sync_copy_func[static_cast<int>(device::DeviceType::kAscend)](dst_device_sync, src_device_sync, stream_id);
  }
  MS_EXCEPTION_IF_NULL(g_sync_copy_func[static_cast<int>(device::DeviceType::kGPU)]);
  return g_sync_copy_func[static_cast<int>(device::DeviceType::kGPU)](dst_device_sync, src_device_sync, stream_id);
}

bool AsyncCopy(const DeviceSyncPtr &dst_device_sync, const DeviceSyncPtr &src_device_sync, size_t stream_id,
               bool keep_host) {
  MS_EXCEPTION_IF_NULL(dst_device_sync);
  MS_EXCEPTION_IF_NULL(src_device_sync);
  if (dst_device_sync->GetDeviceType() == device::DeviceType::kUnknown ||
      src_device_sync->GetDeviceType() == device::DeviceType::kUnknown) {
    MS_LOG(EXCEPTION) << "Invalid device type for device sync:" << dst_device_sync
                      << " type:" << dst_device_sync->GetDeviceType() << " or device sync:" << src_device_sync
                      << " type:" << src_device_sync->GetDeviceType() << " stream id:" << stream_id;
  }
  if (dst_device_sync->GetDeviceType() == device::DeviceType::kCPU &&
      src_device_sync->GetDeviceType() == device::DeviceType::kCPU) {
    MS_EXCEPTION_IF_NULL(g_sync_copy_func[static_cast<int>(device::DeviceType::kCPU)]);
    return g_async_copy_func[static_cast<int>(device::DeviceType::kCPU)](dst_device_sync, src_device_sync, stream_id);
  }
  if (dst_device_sync->GetDeviceType() == device::DeviceType::kAscend ||
      src_device_sync->GetDeviceType() == device::DeviceType::kAscend) {
    MS_EXCEPTION_IF_NULL(g_sync_copy_func[static_cast<int>(device::DeviceType::kAscend)]);
    return g_async_copy_func[static_cast<int>(device::DeviceType::kAscend)](dst_device_sync, src_device_sync,
                                                                            stream_id);
  }
  MS_EXCEPTION_IF_NULL(g_sync_copy_func[static_cast<int>(device::DeviceType::kGPU)]);
  return g_async_copy_func[static_cast<int>(device::DeviceType::kGPU)](dst_device_sync, src_device_sync, stream_id);
}
}  // namespace mindspore
