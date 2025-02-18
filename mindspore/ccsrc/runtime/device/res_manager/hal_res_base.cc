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
#include "runtime/device/res_manager/hal_res_base.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
void *HalResBase::AllocateOffloadMemory(size_t size) const {
  MS_LOG(EXCEPTION) << "Not implemented interface.";
  return nullptr;
}

void HalResBase::FreeOffloadMemory(void *ptr) const {
  MS_LOG(EXCEPTION) << "Not implemented interface.";
  return;
}

bool HalResBase::DestroyEvent(const DeviceEventPtr &event) {
  MS_EXCEPTION_IF_NULL(event);
  if (!event->DestroyEvent()) {
    MS_LOG(ERROR) << "DestroyEvent failed.";
    return false;
  }

  std::lock_guard<std::mutex> lock(device_events_mutex_);
  const auto &iter = std::find(device_events_.begin(), device_events_.end(), event);
  if (iter == device_events_.end()) {
    MS_LOG(ERROR) << "Can't find specified device event.";
    return false;
  }
  (void)device_events_.erase(iter);
  return true;
}

bool HalResBase::DestroyAllEvents() {
  DeviceEventPtrList device_events_inner;
  {
    // Reduce the scopt to prevent deadlock.
    std::lock_guard<std::mutex> lock(device_events_mutex_);
    device_events_inner = device_events_;
    device_events_.clear();
  }
  (void)std::for_each(device_events_inner.begin(), device_events_inner.end(), [this](const auto &event) {
    MS_EXCEPTION_IF_NULL(event);
    if (!event->DestroyEvent()) {
      MS_LOG(ERROR) << "DestroyEvent failed.";
    }
  });
  device_events_.clear();
  return true;
}
}  // namespace device
}  // namespace mindspore
