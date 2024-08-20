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

#include "debug/hooker/acl_data_adapter.h"
#include <string>
#include <vector>
#include <memory>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

constexpr uint32_t kAllKernelNames = 0;
constexpr uint32_t kIsKbyK = 1;

namespace mindspore {
namespace hooker {
void AclDataAdapter::AdaptOnStepBegin(uint32_t device_id, int step_count_num, std::vector<std::string> all_kernel_names,
                                      bool is_kbyk) {
  if (!isLoaded) {
    MS_LOG(WARNING) << "Hook library is not loaded, please check.";
    return;
  }
  auto &loader = HookDynamicLoader::GetInstance();

  auto func_ptr = loader.GetHooker(kHookBegin);
  if (func_ptr != nullptr) {
    auto hooker = reinterpret_cast<HookBeginPtr>(func_ptr);
    MS_LOG(INFO) << "Hook on step begin start.";
    std::map<uint32_t, void *> param_list{};
    TO_MAP(all_kernel_names, kAllKernelNames, param_list);
    TO_MAP(is_kbyk, kIsKbyK, param_list);
    // check if need dump & is_init, generate dump path dir and then init&enable dump
    hooker(device_id, step_count_num, param_list);
  }
}

void AclDataAdapter::AdaptOnStepEnd() {
  if (!isLoaded) {
    MS_LOG(WARNING) << "Hook library is not loaded, please check.";
    return;
  }
  auto &loader = HookDynamicLoader::GetInstance();
  auto func_ptr = loader.GetHooker(kHookEnd);
  if (func_ptr != nullptr) {
    auto hooker = reinterpret_cast<HookEndPtr>(func_ptr);
    std::map<uint32_t, void *> param_list{};
    MS_LOG(INFO) << "Hook on step end start.";
    hooker(param_list);
  }
}

AclDataAdapter::AclDataAdapter() {
  if (common::GetEnv(kMSHookEnable) != kEnable) {
    return;
  }
  auto &loader = HookDynamicLoader::GetInstance();
  isLoaded = loader.LoadLibrary();
}

REGISTER_ADAPTER(device::DeviceType::kAscend, AclDataAdapter);
}  // namespace hooker
}  // namespace mindspore
