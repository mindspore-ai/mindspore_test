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

#include "runtime/runtime_conf/executor_py.h"
#include "runtime/pipeline/pipeline.h"

namespace mindspore {
namespace runtime {

std::shared_ptr<RuntimeExecutor> RuntimeExecutor::instance_ = nullptr;
RuntimeExecutor::RuntimeExecutor() : conf_status_({}) {}

RuntimeExecutor::~RuntimeExecutor() = default;

std::shared_ptr<RuntimeExecutor> RuntimeExecutor::GetInstance() {
  static std::once_flag instance_init_flag_ = {};
  std::call_once(instance_init_flag_, [&]() {
    if (instance_ == nullptr) {
      MS_LOG(DEBUG) << "Create new mindspore RuntimeExecutor";
      instance_ = std::make_shared<RuntimeExecutor>();
    }
  });
  MS_EXCEPTION_IF_NULL(instance_);
  return instance_;
}

void RuntimeExecutor::BindCore(const std::vector<int> &module_bind_core_policy) {
  conf_status_[kThreadBindCore] = true;
  ThreadBindCore::GetInstance().enable_thread_bind_core(module_bind_core_policy);
}

void RuntimeExecutor::BindCoreWithPolicy(const BindCorePolicy &module_bind_core_policy) {
  conf_status_[kThreadBindCore] = true;
  ThreadBindCore::GetInstance().enable_thread_bind_core_with_policy(module_bind_core_policy);
}

void RegRuntimeExecutor(py::module *m) {
  (void)py::class_<RuntimeExecutor, std::shared_ptr<RuntimeExecutor>>(*m, "RuntimeExecutor")
    .def_static("get_instance", &RuntimeExecutor::GetInstance, "Get RuntimeExecutor instance.")
    .def("is_thread_bind_core_configured", &RuntimeExecutor::IsThreadBindCoreConfigured,
         "Check whether thread_bind_core configured.")
    .def("set_thread_bind_core_configured", &RuntimeExecutor::SetThreadBindCoreConfigured,
         "Set thread_bind_core configured.")
    .def("thread_bind_core", &RuntimeExecutor::BindCore, "Bind thread to specific cpus")
    .def("thread_bind_core_with_policy", &RuntimeExecutor::BindCoreWithPolicy,
         "Bind thread to specific cpus with policy generated");
}
}  // namespace runtime
}  // namespace mindspore
