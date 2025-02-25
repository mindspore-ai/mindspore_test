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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_DEVICE_CONTEXT_CONF_OP_PRECISION_CONF_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_DEVICE_CONTEXT_CONF_OP_PRECISION_CONF_H_

#include <memory>
#include <map>
#include <string>
#include "utils/ms_utils.h"
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore {
namespace device {
namespace gpu {
const char kMatmulAllowTf32[] = "matmulAllowTf32";
const char kConvAllowTf32[] = "convAllowTf32";

class GPUOpPrecisionConf {
 public:
  GPUOpPrecisionConf() = default;
  ~GPUOpPrecisionConf() = default;
  GPUOpPrecisionConf(const GPUOpPrecisionConf &) = delete;
  GPUOpPrecisionConf &operator=(const GPUOpPrecisionConf &) = delete;
  static std::shared_ptr<GPUOpPrecisionConf> GetInstance();
  void set_matmul_allow_tf32(const bool &value) {
    matmul_allow_tf32_ = value;
    conf_status_[kMatmulAllowTf32] = true;
  }
  bool matmul_allow_tf32();
  void set_conv_allow_tf32(const bool &value) {
    conv_allow_tf32_ = value;
    conf_status_[kConvAllowTf32] = true;
  }
  bool conv_allow_tf32();
  bool IsMatmulAllowTf32Configured() { return conf_status_.count(kMatmulAllowTf32); }
  bool IsConvAllowTf32Configured() { return conf_status_.count(kConvAllowTf32); }

 private:
  static std::shared_ptr<GPUOpPrecisionConf> inst_context_;
  bool matmul_allow_tf32_{false};
  bool conv_allow_tf32_{true};
  std::map<std::string, bool> conf_status_;
};

void RegGPUOpPrecisionConf(py::module *m);
}  // namespace gpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_DEVICE_CONTEXT_CONF_OP_PRECISION_CONF_H_
