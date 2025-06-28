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

#include "backend/common/backend_common_callback.h"
#include "common/kernel_callback.h"
#include "include/backend/distributed/init.h"
#include "runtime/device/device_address_utils.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/hardware/device_context.h"
#include "runtime/graph_scheduler/actor/kernel_async_infer_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_resize_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_launch_actor.h"
#include "runtime/graph_scheduler/graph_scheduler.h"
#include "runtime/graph_scheduler/pipeline/runtime_pipeline.h"
#include "runtime/graph_scheduler/embedding_cache_scheduler.h"
#include "debug/profiler/profiler.h"

namespace mindspore {
namespace backend {
namespace ms_backend {
namespace {
const auto kSyncDataFromDeviceToHost = "SyncDataFromDeviceToHost";
// The runtime pipeline: InferShape->ResizeKernelMod->LaunchKernel, the latter cannot wait for the former, otherwise
// deadlock may occur.
// 1. infer shape task needs to wait for resize and kernel launch.
// 2. Internally, the resize task only needs to wait for the kernel launch
void WaitAsyncResizeAndLaunchFinish() {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  if (runtime::ActorDispatcher::enable_runtime_multi_pipeline()) {
    const auto &cur_thread_id = std::this_thread::get_id();
    if (runtime::EnableRuntimeNewPipeline()) {
      if (cur_thread_id != runtime::RuntimePipeline::GetInstance().resize_queue()->thread_id() &&
          cur_thread_id != runtime::RuntimePipeline::GetInstance().launch_queue()->thread_id()) {
        runtime::RuntimePipeline::GetInstance().infer_queue()->Wait();
      }

      if (cur_thread_id != runtime::RuntimePipeline::GetInstance().launch_queue()->thread_id()) {
        runtime::RuntimePipeline::GetInstance().resize_queue()->Wait();
      }
    } else {
      if (cur_thread_id != runtime::KernelAsyncResizeActor::GetInstance()->actor_thread_id() &&
          cur_thread_id != runtime::KernelAsyncLaunchActor::GetInstance()->actor_thread_id()) {
        runtime::KernelAsyncInferActor::GetInstance()->Wait();
      }

      if (cur_thread_id != runtime::KernelAsyncLaunchActor::GetInstance()->actor_thread_id()) {
        runtime::KernelAsyncResizeActor::GetInstance()->Wait();
      }
    }
  }

  if (runtime::ActorDispatcher::enable_async_launch_kernel()) {
    if (runtime::EnableRuntimeNewPipeline()) {
      runtime::RuntimePipeline::GetInstance().launch_queue()->Wait();
    } else {
      runtime::KernelAsyncLaunchActor::GetInstance()->Wait();
    }
  }

  PROFILER_END(start_time, runtime::ProfilerModule::kRuntime, runtime::ProfilerEvent::kWaitTaskFinish,
               kSyncDataFromDeviceToHost, false);
}
}  // namespace

// Register a wait callback to kernel::KernelTensor, used to wait runtime async kernel launch task finish when get value
// from device side.
REGISTER_KERNEL_CALLBACK(WaitAsyncResizeAndLaunchFinish);

// Callback function will be set to distributed module.
// This function in invoked when exception happens in this cluster.
void StopRuntimeSchedulerOnException() {
#if ((defined ENABLE_CPU) && (!defined _WIN32) && !defined(__APPLE__))
  MS_LOG(DEBUG) << "Start aborting rpc_node_scheduler.";
  // Abort graph scheduler to avoid hang in rpc communication.
  auto &graph_scheduler = runtime::GraphScheduler::GetInstance();
  if (graph_scheduler.initialized() && graph_scheduler.rpc_node_scheduler() != nullptr) {
    graph_scheduler.rpc_node_scheduler()->Abort();
  }
  MS_LOG(DEBUG) << "End aborting rpc_node_scheduler.";

  MS_LOG(INFO) << "Begin finalize the EmbeddingCacheScheduler.";
  runtime::EmbeddingCacheScheduler::GetInstance().Finalize(false);
  MS_LOG(INFO) << "End finalize the EmbeddingCacheScheduler.";
#endif
}
REGISTER_DISTRIBUTED_CALLBACK(StopRuntimeSchedulerOnException);

void ReleaseKernelRuntime(const std::string &device_name, uint32_t device_id) {
  return device::KernelRuntimeManager::Instance().ReleaseKernelRuntime(device_name, device_id);
}
REGISTER_BACKEND_COMMON_CALLBACK(ReleaseKernelRuntime);

KernelTensorPtr CreateWorkspaceKernelTensor(const device::DeviceContext *device_context, size_t stream_id,
                                            const size_t &workspace_size) {
  return runtime::DeviceAddressUtils::CreateWorkspaceKernelTensor(device_context, stream_id, workspace_size);
}
REGISTER_BACKEND_COMMON_CALLBACK(CreateWorkspaceKernelTensor);
}  // namespace ms_backend
}  // namespace backend
}  // namespace mindspore
