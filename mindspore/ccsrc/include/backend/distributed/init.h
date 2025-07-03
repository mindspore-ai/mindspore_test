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

#ifndef MINDSPORE_CCSRC_DISTRIBUTED_INIT_H_
#define MINDSPORE_CCSRC_DISTRIBUTED_INIT_H_

#include <string>
#include <utility>
#include "include/backend/distributed/collective/collective_manager.h"
#if defined(__linux__) && defined(WITH_BACKEND)
#include "include/backend/distributed/cluster/cluster_context.h"
#else
#include "include/backend/distributed/cluster/dummy_cluster_context.h"
#endif
#include "include/backend/visible.h"

namespace mindspore {
namespace distributed {
// The static methods of MindSpore distributed execution. They can be exported by Pybind.

// Initialize and finalize distributed execution.
BACKEND_COMMON_EXPORT bool Initialize();
BACKEND_COMMON_EXPORT bool Finalize();

// Initialize and finalize the cluster based on MindSpore communication framework.
BACKEND_COMMON_EXPORT bool InitializeCluster();
BACKEND_COMMON_EXPORT bool FinalizeCluster();

// Initialize and finalize collective communication for distributed execution.
BACKEND_COMMON_EXPORT bool InitializeCollective();
BACKEND_COMMON_EXPORT bool FinalizeCollective();

// Set and get whether this process in cluster exits with exception.
BACKEND_COMMON_EXPORT void set_cluster_exit_with_exception();
BACKEND_COMMON_EXPORT bool cluster_exit_with_exception();

BACKEND_COMMON_EXPORT void RegisterCallback(const std::string &name, const std::function<void()> &func);
}  // namespace distributed
}  // namespace mindspore

template <typename Func>
class DistributedCallbackRegister {
 public:
  DistributedCallbackRegister(const std::string &name, Func func) { register_impl(name, std::move(func)); }

 private:
  void register_impl(const std::string &name, std::function<void()> func) {
    mindspore::distributed::RegisterCallback(name, std::move(func));
  }
};

#define REGISTER_DISTRIBUTED_CALLBACK(func) \
  static const DistributedCallbackRegister<std::function<decltype(func)>> g_##func##_callback_register(#func, func)

#endif  // MINDSPORE_CCSRC_DISTRIBUTED_INIT_H_
