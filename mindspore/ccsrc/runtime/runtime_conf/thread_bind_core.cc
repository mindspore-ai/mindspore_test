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

#include "runtime/runtime_conf/thread_bind_core.h"

#ifdef __linux__
#define BIND_CORE
#include <sched.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#endif
#include <algorithm>
#include <sstream>
#include <nlohmann/json.hpp>

#include "utils/ms_utils.h"
#include "utils/ms_context.h"
#include "utils/log_adapter.h"
#include "utils/file_utils.h"
#include "include/backend/distributed/collective/collective_manager.h"

namespace mindspore {
namespace runtime {
namespace {
constexpr char kMainThread[] = "main";
constexpr char kPynativeThread[] = "pynative";
constexpr char kRunTimeThread[] = "runtime";
constexpr char kDataThread[] = "minddata";
constexpr int kMainCoreIdx = 0;
constexpr int kRuntimeCoreIdxStart = 1;
constexpr int kRuntimeCoreIdxEnd = 6;
constexpr int kPynativeCoreIdxStart = 1;
constexpr int kPynativeCoreIdxEnd = 5;
constexpr int kDataCoreIdxStart = 6;
constexpr int kMinimumCorePerProcess = 7;
}  // namespace

int get_device_id() {
  std::string env_device_id = common::GetEnv("DEVICE_ID", "0");
  std::string env_visible_device = common::GetEnv("ASCEND_RT_VISIBLE_DEVICES");
  int device_id;
  if (env_visible_device.empty()) {
    device_id = std::stoi(env_device_id);
  } else {
    std::vector<int> list_visible_device;
    std::stringstream ss(env_visible_device);
    std::string item;
    while (std::getline(ss, item, ',')) {
      list_visible_device.push_back(std::stoi(item));
    }
    std::sort(list_visible_device.begin(), list_visible_device.end());
    device_id = list_visible_device[std::stoi(env_device_id)];
  }
  MS_LOG(INFO) << "The physical device id for this process to bind thread core is " << device_id;
  return device_id;
}

void ThreadBindCore::enable_thread_bind_core(const std::vector<int> &available_cpu_list) {
  if (is_enable_thread_bind_core_) {
    MS_LOG(WARNING)
      << "Thead bind core has already been enabled and will be implemented based on the first binding policy.";
    return;
  }
  cpu_bind_core_policy_ = available_cpu_list;
  is_enable_with_policy = false;
  is_enable_thread_bind_core_ = true;
}

void ThreadBindCore::enable_thread_bind_core_with_policy(const BindCorePolicy &bind_core_policy) {
  if (is_enable_thread_bind_core_) {
    MS_LOG(WARNING)
      << "Thead bind core has already been enabled and will be implemented based on the first binding policy.";
    return;
  }
  process_bind_core_policy_ = bind_core_policy;
  is_enable_with_policy = true;
  is_enable_thread_bind_core_ = true;
}

bool ThreadBindCore::parse_thread_bind_core_policy(const kBindCoreModule &module_name, int device_id) {
  auto it = thread_bind_core_policy_.find(module_name);
  if (it != thread_bind_core_policy_.end()) {
    return true;
  }
  // When automatically enable bind core, device_target CPU and GPU won't bind core based on device to numa affinity.
  if (!is_enable_with_policy) {
    uint32_t local_rank_size = distributed::collective::CollectiveManager::instance()->local_rank_size();
    uint32_t local_rank_id = distributed::collective::CollectiveManager::instance()->local_rank_id();
    int core_per_process = cpu_bind_core_policy_.size() / local_rank_size;
    if (core_per_process < kMinimumCorePerProcess) {
      MS_LOG(WARNING)
        << "CPU can be assigned to each process is less than 7, thread bind core function is not enabled.";
      return false;
    }
    int group_start_core_id = local_rank_id * core_per_process;
    std::vector<int> available_core =
      std::vector<int>(cpu_bind_core_policy_.begin() + group_start_core_id,
                       cpu_bind_core_policy_.begin() + group_start_core_id + core_per_process);

    thread_bind_core_policy_[kBindCoreModule::kMAIN] = {available_core[kMainCoreIdx]};
    thread_bind_core_policy_[kBindCoreModule::kRUNTIME] =
      std::vector<int>(available_core.begin() + kRuntimeCoreIdxStart, available_core.begin() + kRuntimeCoreIdxEnd);
    thread_bind_core_policy_[kBindCoreModule::kPYNATIVE] =
      std::vector<int>(available_core.begin() + kPynativeCoreIdxStart, available_core.begin() + kPynativeCoreIdxEnd);
    thread_bind_core_policy_[kBindCoreModule::kMINDDATA] =
      std::vector<int>(available_core.begin() + kDataCoreIdxStart, available_core.end());
  } else {
    if (process_bind_core_policy_.find(device_id) == process_bind_core_policy_.end()) {
      MS_LOG(WARNING) << "Bind core policy does not include the physical device_id of this process, thread bind core "
                         "function is not enabled. ";
      return false;
    }
    thread_bind_core_policy_[kBindCoreModule::kMAIN] = process_bind_core_policy_[device_id][kMainThread];
    thread_bind_core_policy_[kBindCoreModule::kRUNTIME] = process_bind_core_policy_[device_id][kRunTimeThread];
    thread_bind_core_policy_[kBindCoreModule::kPYNATIVE] = process_bind_core_policy_[device_id][kPynativeThread];
    thread_bind_core_policy_[kBindCoreModule::kMINDDATA] = process_bind_core_policy_[device_id][kDataThread];
  }
  return true;
}

std::vector<int> ThreadBindCore::get_thread_bind_core_list(const kBindCoreModule &module_name) {
  if (!is_enable_thread_bind_core_) {
    MS_LOG(EXCEPTION) << "Cannot get thread bind core list for this module: " << module_name
                      << ", if 'set_cpu_affinity' is turned off.";
  }

  auto status_it = thread_bind_core_status_.find(module_name);
  if (status_it != thread_bind_core_status_.end() && status_it->second) {
    MS_LOG(INFO) << "This module: " << module_name
                 << " has already been assigned a bind core list: " << thread_bind_core_policy_[module_name];
    return thread_bind_core_policy_[module_name];
  }

  int device_id = get_device_id();
  bool res = parse_thread_bind_core_policy(module_name, device_id);
  if (!res) {
    return {};
  }

  auto it = thread_bind_core_policy_.find(module_name);
  if (it == thread_bind_core_policy_.end()) {
    MS_LOG(INFO) << "This module: " << module_name << " has no bind core policy to be assigned.";
    return {};
  }
  MS_LOG(INFO) << "This module: " << module_name << " is assigned a bind core list: " << it->second;
  thread_bind_core_status_[module_name] = true;
  return it->second;
}

void ThreadBindCore::bind_thread_core(const std::vector<int> &cpu_list) {
#if defined(BIND_CORE)
  if (!is_enable_thread_bind_core_) {
    MS_LOG(EXCEPTION) << "Cannot bind core to this thread if 'set_cpu_affinity' is turned off.";
  }

  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  for (const auto &cpu_id : cpu_list) {
    CPU_SET(static_cast<size_t>(cpu_id), &cpuset);
  }

  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  MS_LOG(INFO) << "Enable bind core to core list: " << cpu_list;
#endif  // BIND_CORE
}

bool ThreadBindCore::unbind_thread_core(const std::string &thread_name) { return true; }
}  // namespace runtime
}  // namespace mindspore
