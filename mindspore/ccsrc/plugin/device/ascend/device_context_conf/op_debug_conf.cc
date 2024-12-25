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
#include "plugin/device/ascend/device_context_conf/op_debug_conf.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace ascend {
std::shared_ptr<OpDebugConf> OpDebugConf::inst_context_ = nullptr;

std::shared_ptr<OpDebugConf> OpDebugConf::GetInstance() {
  static std::once_flag inst_context_init_flag_ = {};
  std::call_once(inst_context_init_flag_, [&]() {
    if (inst_context_ == nullptr) {
      MS_LOG(DEBUG) << "Create new mindspore OpDebugConf";
      inst_context_ = std::make_shared<OpDebugConf>();
    }
  });
  MS_EXCEPTION_IF_NULL(inst_context_);
  return inst_context_;
}

void OpDebugConf::set_execute_timeout(uint32_t op_timeout) {
  is_execute_timeout_configured_ = true;
  execute_timeout_ = op_timeout;
}

uint32_t OpDebugConf::execute_timeout() const {
  if (is_execute_timeout_configured_) {
    return execute_timeout_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  uint32_t execute_timeout = ms_context->get_param<uint32_t>(MS_CTX_OP_TIMEOUT);
  return execute_timeout;
}

std::string OpDebugConf::debug_option() const {
  if (!debug_option_.empty()) {
    return debug_option_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto debug_option = ms_context->get_param<std::string>(MS_CTX_OP_DEBUG_OPTION);
  return debug_option;
}

void RegOpDebugConf(py::module *m) {
  (void)py::class_<OpDebugConf, std::shared_ptr<OpDebugConf>>(*m, "AscendOpDebugConf")
    .def_static("get_instance", &OpDebugConf::GetInstance, "Get OpDebugConf instance.")
    .def("set_execute_timeout", &OpDebugConf::set_execute_timeout, "Set Execute Timeout.")
    .def("execute_timeout", &OpDebugConf::execute_timeout, "Get Execute Timeout.")
    .def("set_debug_option", &OpDebugConf::set_debug_option, "Set Debug Option.")
    .def("debug_option", &OpDebugConf::debug_option, "Get Debug Option.")
    .def("is_execute_timeout_configured", &OpDebugConf::IsExecuteTimeoutConfigured, "Is Execute Timeout Configured.")
    .def("is_debug_option_configured", &OpDebugConf::IsDebugOptionConfigured, "Is Debug Option Configured.");
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
