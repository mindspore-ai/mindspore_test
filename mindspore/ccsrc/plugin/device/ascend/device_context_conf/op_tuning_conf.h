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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_TUNING_CONF_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_TUNING_CONF_H_

#include <memory>
#include <string>
#include "runtime/hardware/device_context_manager.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
class OpTuningConf {
 public:
  OpTuningConf() = default;
  ~OpTuningConf() = default;
  OpTuningConf(const OpTuningConf &) = delete;
  OpTuningConf &operator=(const OpTuningConf &) = delete;
  static std::shared_ptr<OpTuningConf> GetInstance();
  void set_jit_compile(const std::string &value) { jit_compile_ = value; }
  void set_aoe_tune_mode(const std::string &tune_mode) { aoe_tune_mode_ = tune_mode; }
  void set_aoe_job_type(const std::string &aoe_config);
  std::string jit_compile() const;
  std::string aoe_job_type() const;
  bool EnableAoeOnline() const;
  bool EnableAoeOffline() const;
  bool IsJitCompileConfigured() const { return !jit_compile_.empty(); }
  bool IsAoeTuneModeConfigured() const { return !aoe_tune_mode_.empty(); }
  bool IsAoeJobTypeConfigured() const { return !aoe_job_type_.empty(); }

 private:
  std::string aoe_tune_mode() const;
  static std::shared_ptr<OpTuningConf> inst_context_;
  std::string jit_compile_{""};
  std::string aoe_tune_mode_{""};
  std::string aoe_job_type_{"2"};
  bool is_aoe_job_type_configured_{false};
};

void RegOpTuningConf(py::module *m);
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_DEVICE_CONTEXT_CONF_OP_TUNING_CONF_H_
