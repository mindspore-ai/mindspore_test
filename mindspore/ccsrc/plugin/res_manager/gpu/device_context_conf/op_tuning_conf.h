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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_DEVICE_CONTEXT_CONF_OP_TUNING_CONF_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_DEVICE_CONTEXT_CONF_OP_TUNING_CONF_H_

#include <memory>
#include <map>
#include <string>
#include "utils/ms_utils.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
namespace device {
namespace gpu {
const char kConvFpropAlgo[] = "convFpropAlgo";
const char kConvWgradAlgo[] = "convWgradAlgo";
const char kConvDgradAlgo[] = "convDgradAlgo";
class GPUOpTuningConf {
 public:
  GPUOpTuningConf() = default;
  ~GPUOpTuningConf() = default;
  GPUOpTuningConf(const GPUOpTuningConf &) = delete;
  GPUOpTuningConf &operator=(const GPUOpTuningConf &) = delete;
  static std::shared_ptr<GPUOpTuningConf> GetInstance();
  void set_conv_fprop_algo(const std::string &value) {
    conv_fprop_algo_ = value;
    conf_status_[kConvFpropAlgo] = true;
  }
  std::string conv_fprop_algo() const;
  void set_conv_wgrad_algo(const std::string &value) {
    conv_wgrad_algo_ = value;
    conf_status_[kConvWgradAlgo] = true;
  }
  std::string conv_wgrad_algo() const;
  void set_conv_dgrad_algo(const std::string &value) {
    conv_dgrad_algo_ = value;
    conf_status_[kConvDgradAlgo] = true;
  }
  std::string conv_dgrad_algo() const;
  bool IsConvFpropAlgoConfigured() { return conf_status_.count(kConvFpropAlgo); }
  bool IsConvWgradAlgoConfigured() { return conf_status_.count(kConvWgradAlgo); }
  bool IsConvDgradAlgoConfigured() { return conf_status_.count(kConvDgradAlgo); }

 private:
  static std::shared_ptr<GPUOpTuningConf> inst_context_;
  std::string conv_fprop_algo_{"normal"};
  std::string conv_wgrad_algo_{"normal"};
  std::string conv_dgrad_algo_{"normal"};
  std::map<std::string, bool> conf_status_;
};

void RegGPUOpTuningConf(py::module *m);

}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_DEVICE_CONTEXT_CONF_OP_TUNING_CONF_H_
