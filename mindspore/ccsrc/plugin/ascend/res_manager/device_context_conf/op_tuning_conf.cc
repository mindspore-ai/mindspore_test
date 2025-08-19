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
#include "plugin/ascend/res_manager/device_context_conf/op_tuning_conf.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace ascend {
std::shared_ptr<OpTuningConf> OpTuningConf::inst_context_ = nullptr;

std::shared_ptr<OpTuningConf> OpTuningConf::GetInstance() {
  static std::once_flag inst_context_init_flag_ = {};
  std::call_once(inst_context_init_flag_, [&]() {
    if (inst_context_ == nullptr) {
      MS_LOG(DEBUG) << "Create new mindspore OpTuningConf";
      inst_context_ = std::make_shared<OpTuningConf>();
    }
  });
  MS_EXCEPTION_IF_NULL(inst_context_);
  return inst_context_;
}

std::string OpTuningConf::jit_compile() const {
  if (!jit_compile_.empty()) {
    return jit_compile_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto jit_compile = ms_context->get_param<std::string>(MS_CTX_ENABLE_JIT_COMPILE);
  return jit_compile;
}

void OpTuningConf::set_aoe_job_type(const std::string &aoe_config) {
  is_aoe_job_type_configured_ = true;
  aoe_job_type_ = aoe_config;
}

std::string OpTuningConf::aoe_tune_mode() const {
  if (!aoe_tune_mode_.empty()) {
    return aoe_tune_mode_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto aoe_tune_mode = ms_context->get_param<std::string>(MS_CTX_AOE_TUNE_MODE);
  return aoe_tune_mode;
}

std::string OpTuningConf::aoe_job_type() const {
  if (is_aoe_job_type_configured_) {
    return aoe_job_type_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto aoe_job_type = ms_context->get_param<std::string>(MS_CTX_AOE_JOB_TYPE);
  return aoe_job_type;
}

bool OpTuningConf::EnableAoeOnline() const {
  std::string mode = aoe_tune_mode();
  return mode == "online";
}

bool OpTuningConf::EnableAoeOffline() const {
  std::string mode = aoe_tune_mode();
  return mode == "offline";
}

void RegOpTuningConf(py::module *m) {
  (void)py::class_<OpTuningConf, std::shared_ptr<OpTuningConf>>(*m, "AscendOpTuningConf")
    .def_static("get_instance", &OpTuningConf::GetInstance, "Get OpTuningConf instance.")
    .def("set_jit_compile", &OpTuningConf::set_jit_compile, "Set Jit Compile.")
    .def("set_aoe_tune_mode", &OpTuningConf::set_aoe_tune_mode, "Set Aoe Tune Mode.")
    .def("set_aoe_job_type", &OpTuningConf::set_aoe_job_type, "Set Aoe Job Type.")
    .def("is_jit_compile_configured", &OpTuningConf::IsJitCompileConfigured, "Is Jit Compile Configured.")
    .def("is_aoe_tune_mode_configured", &OpTuningConf::IsAoeTuneModeConfigured, "Is Aoe Tune Mode Configured.")
    .def("is_aoe_job_type_configured", &OpTuningConf::IsAoeJobTypeConfigured, "Is Aoe Job Type Configured.")
    .def("jit_compile", &OpTuningConf::jit_compile, "Get Jit Compile.")
    .def("aoe_job_type", &OpTuningConf::aoe_job_type, "Get Aoe Tune Mode.")
    .def("aoe_tune_mode", &OpTuningConf::aoe_tune_mode, "Get Aoe Job Type.")
    .def("is_aclnn_cache_configured", &OpTuningConf::IsAclnnCacheConfigured, "Is Aclnn Cache Configured.")
    .def("set_aclnn_global_cache", &OpTuningConf::SetAclnnGlobalCache, "Set Aclnn Global Cache Mode.")
    .def("set_cache_queue_length", &OpTuningConf::SetAclnnCacheQueueLength, "Set Aclnn Cache Queue Length.");
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
