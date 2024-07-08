/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "pybind_api/hal/comm_handle_py.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/pipeline/task/device_task.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/ms_context.h"
#include "include/common/pybind_api/api_register.h"
#include "pipeline/pynative/forward/forward_task.h"
#include "pipeline/pynative/pynative_utils.h"

namespace mindspore {
namespace hal {
CommHandlePy::~CommHandlePy() {
  if (comm_handle_ == nullptr) {
    return;
  }

  pynative::DispatchOp(std::make_shared<pynative::PassthroughFrontendTask>([comm_handle = comm_handle_]() {
    auto destroy_fn = [comm_handle]() {
      runtime::OpExecutor::DispatchLaunchTask([device_ctx = comm_handle->device_ctx(), event = comm_handle->event()]() {
        MS_EXCEPTION_IF_NULL(device_ctx);
        if (event == nullptr) {
          // event is nullptr in gpu and cpu
          return;
        }

        MS_LOG(DEBUG) << "DestroyEvent, event:" << event;
        if (device_ctx != nullptr && device_ctx->initialized()) {
          device_ctx->device_res_manager_->DestroyEvent(event);
        }
      });
    };
    if (!runtime::OpExecutor::NeedSync()) {
      runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
        std::make_shared<runtime::PassthroughNoWaitDeviceTask>(destroy_fn));
    } else {
      destroy_fn();
    }
  }));
}

void CommHandlePy::Wait() {
  if (comm_handle_ == nullptr) {
    return;
  }

  MS_EXCEPTION_IF_NULL(device_ctx_);
  auto cur_stream_id = device_ctx_->device_res_manager_->GetCurrentStreamId();
  // Wait event async.
  pynative::DispatchOp(
    std::make_shared<pynative::PassthroughFrontendTask>([cur_stream_id, comm_handle = comm_handle_]() {
      auto wait_fn = [comm_handle, cur_stream_id]() {
        runtime::OpExecutor::DispatchLaunchTask([comm_handle, cur_stream_id]() {
          MS_EXCEPTION_IF_NULL(comm_handle);
          comm_handle->WaitDeviceEvent(cur_stream_id);
        });

        comm_handle->ReleaseMultiStreamEvent(cur_stream_id);
      };
      if (!runtime::OpExecutor::NeedSync()) {
        runtime::OpExecutor::GetInstance().PushSimpleOpRunTask(
          std::make_shared<runtime::PassthroughNoWaitDeviceTask>(wait_fn));
      } else {
        wait_fn();
      }
    }));
}

void RegCommHandle(py::module *m) {
  (void)py::class_<CommHandlePy, std::shared_ptr<CommHandlePy>>(*m, "CommHandle")
    .def(py::init<>())
    .def("wait", &CommHandlePy::Wait);
}
}  // namespace hal
}  // namespace mindspore
