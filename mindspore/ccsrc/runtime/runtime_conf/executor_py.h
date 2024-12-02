/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_RUNTIME_CONF_EXECUTOR_PY_H
#define MINDSPORE_CCSRC_RUNTIME_RUNTIME_CONF_EXECUTOR_PY_H
#include <map>
#include <string>
#include <vector>
#include <memory>

#include "runtime/runtime_conf/thread_bind_core.h"
#include "include/common/pybind_api/api_register.h"

namespace mindspore {
namespace runtime {
constexpr char kThreadBindCore[] = "thread_bind_core";
class BACKEND_EXPORT RuntimeExecutor {
 public:
  RuntimeExecutor();
  ~RuntimeExecutor();
  RuntimeExecutor(const RuntimeExecutor &) = delete;
  RuntimeExecutor &operator=(const RuntimeExecutor &) = delete;
  static std::shared_ptr<RuntimeExecutor> GetInstance();

  bool IsThreadBindCoreConfigured() { return conf_status_.count(kThreadBindCore); }

  void BindThreadCpu(const std::map<int, std::vector<int>> &bind_cpu_policy, bool custom_plicy_flag);

 private:
  static std::shared_ptr<RuntimeExecutor> instance_;
  std::map<std::string, bool> conf_status_;
};
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_RUNTIME_CONF_EXECUTOR_PY_H
