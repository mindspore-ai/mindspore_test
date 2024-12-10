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
static const std::vector<std::string> kPynativeThreadName = {"frontend_queue", "backend_queue", "launch_queue",
                                                             "bprop_queue"};
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

bool ThreadBindCore::parse_thread_bind_core_policy(const std::string &thread_name, int device_id) {
  auto it = thread_bind_core_policy_.find(thread_name);
  if (it != thread_bind_core_policy_.end()) {
    return true;
  }
  // When automatically enable bind core, device_target CPU and GPU won't bind core based on device to numa affinity.
  if (!is_enable_with_policy) {
    uint32_t local_rank_size = distributed::collective::CollectiveManager::instance()->local_rank_size();
    uint32_t local_rank_id = distributed::collective::CollectiveManager::instance()->local_rank_id();
    int core_per_process = cpu_bind_core_policy_.size() / local_rank_size;
    if (core_per_process < 7) {
      MS_LOG(WARNING)
        << "CPU can be assigned to each process is less than 7, thread bind core function is not enabled.";
      return false;
    }
    int group_start_core_id = local_rank_id * core_per_process;
    std::vector<int> available_core =
      std::vector<int>(cpu_bind_core_policy_.begin() + group_start_core_id,
                       cpu_bind_core_policy_.begin() + group_start_core_id + core_per_process);

    thread_bind_core_policy_[kMainThread] = {available_core[0]};
    thread_bind_core_policy_[kRunTimeThread] = std::vector<int>(available_core.begin() + 1, available_core.begin() + 6);
    thread_bind_core_policy_[kDataThread] = std::vector<int>(available_core.begin() + 6, available_core.end());
    for (size_t i = 0; i < kPynativeThreadName.size(); i++) {
      thread_bind_core_policy_[kPynativeThreadName[i]] = {
        std::vector<int>(available_core.begin() + 1, available_core.begin() + 5)[i]};
    }
  } else {
    if (process_bind_core_policy_.find(device_id) == process_bind_core_policy_.end()) {
      MS_LOG(WARNING) << "Bind core policy does not include the physical device_id of this process, thread bind core "
                         "function is not enabled. ";
      return false;
    }
    thread_bind_core_policy_[kMainThread] = process_bind_core_policy_[device_id][kMainThread];
    thread_bind_core_policy_[kRunTimeThread] = process_bind_core_policy_[device_id][kRunTimeThread];
    thread_bind_core_policy_[kDataThread] = process_bind_core_policy_[device_id][kDataThread];
    for (size_t i = 0; i < kPynativeThreadName.size(); i++) {
      thread_bind_core_policy_[kPynativeThreadName[i]] = {process_bind_core_policy_[device_id][kPynativeThread][i]};
    }
  }
  return true;
}

void ThreadBindCore::bind_thread_core(const std::string &thread_name) {
#if defined(BIND_CORE)
  if (!is_enable_thread_bind_core_) {
    return;
  }
  auto status_it = thread_bind_core_status_.find(thread_name);
  if (status_it != thread_bind_core_status_.end() && status_it->second) {
    MS_LOG(INFO) << "This thead: " << thread_name << " has already bond core.";
    return;
  }

  int device_id = get_device_id();
  bool res = parse_thread_bind_core_policy(thread_name, device_id);
  if (!res) {
    return;
  }

  auto it = thread_bind_core_policy_.find(thread_name);
  auto &thread_bind_cpu_vec = it->second;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  for (const auto &cpu_id : thread_bind_cpu_vec) {
    // auto bind_cpu_id = group_start_cpu_id + cpu_id;
    CPU_SET(static_cast<size_t>(cpu_id), &cpuset);
  }

  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  thread_bind_core_status_[thread_name] = true;
  MS_LOG(INFO) << "Enable bind core for thread: " << thread_name << " to core list: " << thread_bind_cpu_vec;
#endif  // BIND_CORE
}

bool ThreadBindCore::unbind_thread_core(const std::string &thread_name) { return true; }

std::vector<int> ThreadBindCore::get_thread_bind_core_list(const std::string &thread_name) {
  if (!is_enable_thread_bind_core_) {
    return {};
  }
  int device_id = get_device_id();
  bool res = parse_thread_bind_core_policy(thread_name, device_id);
  if (!res) {
    return {};
  }

  auto it = thread_bind_core_policy_.find(thread_name);

  if (it == thread_bind_core_policy_.end()) {
    MS_LOG(INFO) << "Thread: " << thread_name << " has no bind policy, bind failed";
    return {};
  }
  MS_LOG(INFO) << "Enable bind core for thread: " << thread_name
               << ", get core list: " << thread_bind_core_policy_[thread_name];
  return thread_bind_core_policy_[thread_name];
}
}  // namespace runtime
}  // namespace mindspore
