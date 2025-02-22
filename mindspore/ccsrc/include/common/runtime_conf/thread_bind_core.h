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

#ifndef MINDSPORE_CCSRC_INCLUDE_COMMON_RUNTIME_CONF_THREAD_BIND_CORE_H_
#define MINDSPORE_CCSRC_INCLUDE_COMMON_RUNTIME_CONF_THREAD_BIND_CORE_H_

#include <string>
#include <memory>
#include <map>
#include <vector>
#include <set>
#include <mutex>
#include <iostream>

#include "include/backend/visible.h"
#include "mindapi/base/macros.h"
#include "utils/log_adapter.h"
#include "include/common/visible.h"

namespace mindspore {
namespace runtime {
using BindCorePolicy = std::map<int, std::map<std::string, std::vector<int>>>;
enum kBindCoreModule : int { kMAIN = 0, kRUNTIME, kPYNATIVE, kMINDDATA, kBATCHLAUNCH };

class COMMON_EXPORT ThreadBindCore {
 public:
  static ThreadBindCore &GetInstance() {
    static ThreadBindCore instance;
    return instance;
  }
  void enable_thread_bind_core(const std::vector<int> &available_cpu_list);
  void enable_thread_bind_core_with_policy(const BindCorePolicy &bind_core_policy);
  bool parse_thread_bind_core_policy(const kBindCoreModule &module_name, uint32_t device_id);
  std::vector<int> get_thread_bind_core_list(const kBindCoreModule &module_name);
  void bind_thread_core(const std::vector<int> &cpu_list);
  bool unbind_thread_core(const std::string &thread_name);
  bool is_enable_thread_bind_core_{false};

 private:
  BindCorePolicy process_bind_core_policy_;
  std::vector<int> cpu_bind_core_policy_;
  std::map<kBindCoreModule, std::vector<int>> thread_bind_core_policy_;
  std::map<kBindCoreModule, bool> thread_bind_core_status_;
  bool is_enable_with_policy{false};
  std::mutex mtx_;
  ThreadBindCore() = default;
  ~ThreadBindCore() = default;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_INCLUDE_COMMON_RUNTIME_CONF_THREAD_BIND_CORE_H_
