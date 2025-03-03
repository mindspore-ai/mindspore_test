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

#include "common/kernel_callback.h"
#ifndef BUILD_LITE
#include "runtime/graph_scheduler/actor/kernel_async_infer_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_resize_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_launch_actor.h"
#endif

namespace mindspore {
namespace {
void WaitAsyncResizeAndLaunchFinish() {
#ifndef BUILD_LITE
  if (runtime::ActorDispatcher::enable_runtime_multi_pipeline()) {
    const auto &cur_thread_id = std::this_thread::get_id();
    if (cur_thread_id != runtime::KernelAsyncResizeActor::GetInstance()->actor_thread_id() &&
        cur_thread_id != runtime::KernelAsyncLaunchActor::GetInstance()->actor_thread_id()) {
      runtime::KernelAsyncInferActor::GetInstance()->Wait();
    }

    if (cur_thread_id != runtime::KernelAsyncLaunchActor::GetInstance()->actor_thread_id()) {
      runtime::KernelAsyncResizeActor::GetInstance()->Wait();
    }
  }

  if (runtime::ActorDispatcher::enable_async_launch_kernel()) {
    runtime::KernelAsyncLaunchActor::GetInstance()->Wait();
  }
#endif
}
}  // namespace
REGISTER_KERNEL_CALLBACK(WaitAsyncResizeAndLaunchFinish);
}  // namespace mindspore
