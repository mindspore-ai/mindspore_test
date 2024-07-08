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

#ifndef MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_COMM_HANDLE_H_
#define MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_COMM_HANDLE_H_

#include <memory>
#include "include/common/visible.h"
#include "runtime/hardware/device_context.h"
#include "ir/device_event.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
class BACKEND_EXPORT CommHandle {
 public:
  CommHandle() = default;
  explicit CommHandle(const device::DeviceContext *device_ctx) : device_ctx_(device_ctx) {}
  ~CommHandle() = default;

  DeviceEventPtr CreateEvent();

  void RecordEvent(size_t stream_id);

  void UpdateTaskId(size_t stream_id);

  void WaitDeviceEvent(size_t cur_stream_id);

  void ReleaseMultiStreamEvent(size_t cur_stream_id);

  DeviceEventPtr event() const { return event_; }

  const device::DeviceContext *device_ctx() const { return device_ctx_; }

 private:
  DeviceEventPtr event_{nullptr};
  // Transfer between multi-stage pipelines
  std::shared_ptr<int64_t> task_id_on_stream_ = std::make_shared<int64_t>(0L);
  std::shared_ptr<size_t> record_stream_id_ = std::make_shared<size_t>(0);

  const device::DeviceContext *device_ctx_;
};

using CommHandlePtr = std::shared_ptr<CommHandle>;

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_MINDSPORE_CCSRC_KERNEL_PYBOOST_COMM_HANDLE_H_
