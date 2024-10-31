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

#include "utils/llm_manager.h"

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
#include "utils/log_adapter.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace {
constexpr char kPynativeThread[] = "pynative";
constexpr char kRunTimeThread[] = "runtime";
constexpr char kProcessCpuSize[] = "each_process_cpu_core_size";
static const std::vector<std::string> kPynativeThreadName = {"frontend_queue", "backend_queue", "launch_queue"};
static const std::vector<std::string> kRuntimeThreadName = {"runtime_scheduler_actor", "runtime_launch_actor"};
}  // namespace

LLMManager::LLMManager() { get_thread_bind_policy(); }

LLMManager &LLMManager::GetInstance() noexcept {
  static LLMManager instance;
  return instance;
}

tensor::TensorDataPtr LLMManager::get_graph_input(const std::string &name) {
  auto it = graph_inputs_map_.find(name);
  if (it == graph_inputs_map_.end()) {
    return nullptr;
  }
  return it->second;
}

void LLMManager::add_graph_input(const std::string &name, tensor::TensorDataPtr tensor) {
  graph_inputs_map_[name] = tensor;
}

void LLMManager::reset_graph_inputs() { graph_inputs_map_.clear(); }

void LLMManager::add_force_resize_kernel(const std::string &kernel_name) {
  force_resize_kernel_set_.insert(kernel_name);
  force_resize_kernel_ = true;
}

bool LLMManager::need_force_resize(const std::string &kernel_name) {
  if (!force_resize_kernel_) {
    return false;
  }
  auto it = std::find(force_resize_kernel_set_.begin(), force_resize_kernel_set_.end(), kernel_name);
  return it != force_resize_kernel_set_.end();
}

void LLMManager::bind_thread_core(const std::string &thread_name) {
#if defined(BIND_CORE)
  auto status_it = thread_bind_status_.find(thread_name);
  if (status_it != thread_bind_status_.end() && status_it->second) {
    // already bind thread, skip
    return;
  }

  auto it = thread_bind_policy_.find(thread_name);
  if (it == thread_bind_policy_.end()) {
    // no thread bind policy
    MS_LOG(WARNING) << "thread: " << thread_name << " has no bind policy, bind failed";
    return;
  }

  auto thread_bind_core_vec = it->second;
  auto rank_id = std::stoi(common::GetEnv("RANK_ID", "0"));
  unsigned group_start_core_id = rank_id * group_core_size_;
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);

  for (const auto &core_id : thread_bind_core_vec) {
    auto bind_core_id = group_start_core_id + core_id;
    CPU_SET(bind_core_id, &cpuset);
    MS_LOG(INFO) << "bind thread: " << thread_name << " to core: " << bind_core_id;
  }

  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
  thread_bind_status_[thread_name] = true;

  return;
#endif  // BIND_CORE
}

bool LLMManager::unbind_threads(const std::string &thread_name) { return true; }

// get thread_bind_policy_ from env
void LLMManager::get_thread_bind_policy() {
  auto thread_bind_env_str = common::GetEnv("MS_ENABLE_NUMA");
  // custom bind policy by json
  auto real_filename = FileUtils::GetRealPath(thread_bind_env_str.c_str());
  if (!real_filename.has_value()) {
    MS_LOG(WARNING) << "Not custom numa policy";
    return;
  }

  std::ifstream json_file(real_filename.value(), std::ifstream::in);
  auto custom_bind_policy = nlohmann::json::parse(json_file);
  auto rank_id = std::stoi(common::GetEnv("RANK_ID", "0"));
  std::string rank_info = "rank_" + std::to_string(rank_id);

  if (custom_bind_policy.contains(kProcessCpuSize)) {
    group_core_size_ = custom_bind_policy[kProcessCpuSize];
    rank_info = "rank_0";
  }
  MS_LOG(INFO) << "rank_info: " << rank_info << " custom_bind_policy: " << custom_bind_policy.dump();

  if (!custom_bind_policy.contains(rank_info)) {
    MS_LOG(WARNING) << "The " << rank_info << " is not existed, custom numa policy failed";
    return;
  }

  if (custom_bind_policy[rank_info].contains(kPynativeThread)) {
    auto pynative_cpus = custom_bind_policy[rank_info][kPynativeThread].get<std::vector<int64_t>>();
    if (pynative_cpus.size() == kPynativeThreadName.size()) {
      for (size_t i = 0; i < pynative_cpus.size(); i++) {
        thread_bind_policy_[kPynativeThreadName[i]] = {pynative_cpus[i]};
      }
    }
  }

  if (custom_bind_policy[rank_info].contains(kRunTimeThread)) {
    auto runtime_cpus = custom_bind_policy[rank_info][kRunTimeThread].get<std::vector<int64_t>>();
    if (runtime_cpus.size() == kRuntimeThreadName.size()) {
      for (size_t i = 0; i < runtime_cpus.size(); i++) {
        thread_bind_policy_[kRuntimeThreadName[i]] = {runtime_cpus[i]};
      }
    }
  }
}
}  // namespace mindspore
