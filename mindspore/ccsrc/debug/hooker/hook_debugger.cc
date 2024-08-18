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

#include "debug/hooker/hook_debugger.h"
#include <string>
#include "debug/hooker/adapter.h"

namespace mindspore {
namespace hooker {
HookDebugger &HookDebugger::GetInstance() {
  static HookDebugger hook_debugger;
  return hook_debugger;
}

bool HookDebugger::IsHookerEnabled() { return common::GetEnv(kMSHookEnable) == kEnable; }

void HookDebugger::HookOnStepBegin(uint32_t device_id, const std::vector<KernelGraphPtr> &graphs, int step_count,
                                   bool is_dataset_sink, bool is_kbyk) {
  if (!IsHookerEnabled()) {
    MS_LOG(WARNING) << "Dump Hook is not enabled, please set MS_HOOK_ENABLE";
    return;
  }

  std::vector<std::string> all_kernel_names;
  for (const auto &graph : graphs) {
    auto all_kernels = graph->execution_order();
    std::for_each(all_kernels.begin(), all_kernels.end(),
                  [&](const auto &k) { all_kernel_names.push_back(k->fullname_with_scope()); });
  }

  auto step_count_num = 0;
  step_count_num = step_count;
  if (step_count == 1 && is_dataset_sink == 1) {
    step_count_num = 0;
  }
  if (!graphs.empty()) {
    auto graph = graphs[0];
    is_dataset_sink = graph->IsDatasetGraph();
  }
  auto registered_adapter = hooker::AdapterManager::Instance().GetAdapterForBackend(device::DeviceType::kAscend);
  if (registered_adapter != nullptr) {
    registered_adapter->AdaptOnStepBegin(device_id, step_count_num, all_kernel_names, is_kbyk);
  } else {
    MS_LOG(WARNING) << "Ascend Adapter is not found! Hook Dump not validate!";
  }
}

void HookDebugger::HookOnStepEnd() {
  if (!IsHookerEnabled()) {
    MS_LOG(WARNING) << "Dump Hook is not enabled, please set MS_HOOK_ENABLE";
    return;
  }

  auto registered_adapter = hooker::AdapterManager::Instance().GetAdapterForBackend(device::DeviceType::kAscend);
  if (registered_adapter != nullptr) {
    registered_adapter->AdaptOnStepEnd();
  } else {
    MS_LOG(WARNING) << "Ascend Adapter is not found! Hook Dump not validate!";
  }
}
}  // namespace hooker
}  // namespace mindspore
