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

#include <memory>
#include "mindspore/ccsrc/pyboost/comm_handle.h"
#include "mindspore/ccsrc/pyboost/comm_utils.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pynative/task/device_task.h"
#include "runtime/device/res_manager/hal_res_manager.h"
namespace mindspore {
namespace kernel {
namespace pyboost {
CommHandle::~CommHandle() {
  if (event_ == nullptr) {
    return;
  }
  if (device_ctx_ != nullptr && device_ctx_->initialized()) {
    device_ctx_->device_res_manager_->DestroyEvent(event_);
    MS_LOG(DEBUG) << "DestoryEvent done, event: " << event_;
  }
}

DeviceEventPtr CommHandle::CreateEvent() {
  MS_EXCEPTION_IF_NULL(device_ctx_);
  device_ctx_->device_res_manager_->BindDeviceToCurrentThread(false);
  event_ = device_ctx_->device_res_manager_->CreateEventWithFlag(false, false);
  return event_;
}

void CommHandle::RecordEvent(size_t stream_id) {
  // Call this function in device thread
  event_->RecordEvent(stream_id);
}

void CommHandle::UpdateTaskId(size_t stream_id) {
  auto &controller =
    device::HalResManager::GetInstance().GetMultiStreamController(device_ctx_->device_context_key().device_name_);
  controller->Refresh();
  auto task_id = controller->LaunchTaskIdOnStream(stream_id);
  *task_id_on_stream_ = task_id;
  *record_stream_id_ = stream_id;
}

void CommHandle::WaitDeviceEvent(size_t cur_stream_id) {
  MS_EXCEPTION_IF_NULL(device_ctx_);
  if (event_ == nullptr) {
    return;
  }

  auto cur_stream_ptr = device_ctx_->device_res_manager_->GetStream(cur_stream_id);
  event_->set_wait_stream(cur_stream_ptr);
  event_->WaitEventWithoutReset();
}

void CommHandle::ReleaseMultiStreamEvent(size_t cur_stream_id) {
  MS_EXCEPTION_IF_NULL(device_ctx_);
  if (event_ == nullptr) {
    return;
  }

  MS_LOG(DEBUG) << "WaitEvent wait stream id:" << cur_stream_id << ", record_stream_id:" << *record_stream_id_
                << ", event:" << event_ << ", task_id_on_stream:" << *task_id_on_stream_;
  // Release cross stream memory event, mark record_stream_id is use stream id, wait stream id is memory stream
  // id.
  (void)device::HalResManager::GetInstance()
    .GetMultiStreamController(device_ctx_->device_context_key().device_name_)
    ->WaitEvent(*task_id_on_stream_, *record_stream_id_, cur_stream_id);
}

void WaitTaskFunc(CommHandlePtr comm_handle) {
  MS_EXCEPTION_IF_NULL(comm_handle);
  auto cur_stream_id = comm_handle->device_ctx()->device_res_manager_->GetCurrentStreamId();
  auto wait_fn = [cur_stream_id, comm_handle]() {
    runtime::OpExecutor::DispatchLaunchTask(
      [cur_stream_id, comm_handle]() { comm_handle->WaitDeviceEvent(cur_stream_id); });

    comm_handle->ReleaseMultiStreamEvent(cur_stream_id);
  };
  if (!runtime::OpExecutor::NeedSync()) {
    runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
      std::make_shared<runtime::PassthroughNoWaitDeviceTask>(wait_fn));
  } else {
    wait_fn();
  }
}

void CommHandle::Wait() {
  // Wait event async.
  WaitTaskFunc(shared_from_this());
  return;
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
