/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_MULTI_STREAM_CONTROLLER_HEADER_H
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_MULTI_STREAM_CONTROLLER_HEADER_H

#include <cstdint>

#include <functional>
#include <list>
#include <memory>
#include <mutex>
#include <vector>
#include <unordered_map>
#include <utility>

#include "async/spinlock.h"
#include "runtime/hardware_abstract/event/device_event.h"
#include "runtime/hardware_abstract/device_context/device_context.h"
#include "runtime/hardware_abstract/visible.h"

namespace mindspore {
namespace device {
class TaskIdOnStreamManager;
using TaskIdOnStreamManagerPtr = std::shared_ptr<TaskIdOnStreamManager>;

class EventPool;
using EventPoolPtr = std::shared_ptr<EventPool>;

class RUNTIME_HARDWARE_EXPORT MultiStreamController {
 public:
  explicit MultiStreamController(DeviceResManager *device_res_base);

  MultiStreamController(const MultiStreamController &) = delete;
  MultiStreamController &operator=(const MultiStreamController &) = delete;
  MultiStreamController(const MultiStreamController &&) = delete;

  ~MultiStreamController() = default;

  void Refresh();

  bool UpdateTaskIdOnStream(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id);

  int64_t QueryTaskIdOnStream(uint32_t user_stream_id, uint32_t memory_stream_id);

  int64_t LaunchTaskIdOnStream(uint32_t stream_id);
  int64_t GetTaskIdOnStream(uint32_t stream_id);

  std::mutex &GetStreamMutex(size_t stream_id);

  // memory_stream_addresses pair : memory_stream_id, address.
  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                   const std::vector<std::pair<uint32_t, void *>> &memory_stream_addresses,
                   const DeviceEventPtr &input_event = nullptr);
  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id);
  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id);
  bool DispatchRecordWaitEvent(uint32_t user_stream_id, uint32_t memory_stream_id);

  bool SyncStream(size_t stream_id);
  bool SyncAllStreams();
  bool SyncNotDefaultStreams();

  bool WaitMultiStream(size_t wait_stream_id);

 protected:
  TaskIdOnStreamManagerPtr task_id_on_stream_manager_;
  std::unordered_map<uint32_t, std::mutex> stream_mutexes_;
  EventPoolPtr event_pool_;

  DeviceResManager *device_res_base_;
  SpinLock lock_;
};
using MultiStreamControllerPtr = std::shared_ptr<MultiStreamController>;
}  // namespace device
}  // namespace mindspore
#endif
