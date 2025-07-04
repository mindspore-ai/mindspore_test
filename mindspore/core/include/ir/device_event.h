/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_IR_DEVICE_EVENT_H
#define MINDSPORE_CORE_IR_DEVICE_EVENT_H

#include <memory>
#include <vector>

namespace mindspore {
class DeviceEvent {
 public:
  virtual ~DeviceEvent() = default;
  virtual bool IsReady() const = 0;
  virtual void WaitEvent() = 0;
  virtual bool WaitEvent(uint32_t stream_id) = 0;
  virtual void WaitEventWithoutReset() = 0;
  virtual void WaitEventWithoutReset(uint32_t stream_id) {}
  virtual void ResetEvent() {}
  virtual void ResetEvent(uint32_t stream_id) {}
  virtual void RecordEvent() = 0;
  virtual void RecordEvent(uint32_t stream_id) = 0;
  virtual bool NeedWait() = 0;
  virtual void SyncEvent() = 0;
  virtual bool QueryEvent() = 0;
  virtual void ElapsedTime(float *cost_time, const DeviceEvent *other) = 0;
  virtual bool DestroyEvent() = 0;
  virtual void set_wait_stream(void *stream) = 0;
  virtual void set_record_stream(void *stream) = 0;
};
using DeviceEventPtr = std::shared_ptr<DeviceEvent>;
using DeviceEventPtrList = std::vector<DeviceEventPtr>;
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_DEVICE_EVENT_H
