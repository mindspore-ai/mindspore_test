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

#include "backend/ge_backend/dump/acl_data_adapter.h"
#include <string>
#include <vector>
#include <memory>
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace dump {

constexpr uint32_t kAllKernelNames = 0;
constexpr uint32_t kIsKbyK = 1;

void AclDataAdapter::AdaptOnStepBegin(uint32_t device_id, int step_count_num,
                                      std::vector<std::string> &&all_kernel_names, bool is_kbyk) {
  if (!isLoaded_) {
    MS_LOG(WARNING) << "Hook library is not loaded, please check.";
    return;
  }
  auto &loader = mindspore::datadump::HookDynamicLoader::GetInstance();

  auto func_ptr = loader.GetHooker(mindspore::datadump::kHookBegin);
  if (func_ptr != nullptr) {
    auto hooker = reinterpret_cast<HookBeginPtr>(func_ptr);
    MS_LOG(INFO) << "Hook on step begin start.";
    std::map<uint32_t, void *> param_list{};
    param_list[kAllKernelNames] = static_cast<void *>(&all_kernel_names);
    param_list[kIsKbyK] = static_cast<void *>(&is_kbyk);
    // check if need dump & is_init, generate dump path dir and then init&enable dump
    hooker(device_id, step_count_num, param_list);
  }
}

void AclDataAdapter::AdaptOnStepEnd() {
  if (!isLoaded_) {
    MS_LOG(WARNING) << "Hook library is not loaded, please check.";
    return;
  }
  auto &loader = mindspore::datadump::HookDynamicLoader::GetInstance();
  auto func_ptr = loader.GetHooker(mindspore::datadump::kHookEnd);
  if (func_ptr != nullptr) {
    auto hooker = reinterpret_cast<HookEndPtr>(func_ptr);
    std::map<uint32_t, void *> param_list{};
    MS_LOG(INFO) << "Hook on step end start.";
    hooker(param_list);
  }
}

void AclDataAdapter::Load() {
  if (common::GetEnv(kMSHookEnable) != kEnable || isLoaded_) {
    return;
  }
  auto &loader = mindspore::datadump::HookDynamicLoader::GetInstance();
  isLoaded_ = loader.LoadLibrary();
}

REGISTER_ADAPTER(device::DeviceType::kAscend, AclDataAdapter);
}  // namespace dump
}  // namespace mindspore
