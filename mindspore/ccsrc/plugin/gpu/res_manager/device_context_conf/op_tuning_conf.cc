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
#include "plugin/gpu/res_manager/device_context_conf/op_tuning_conf.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace gpu {
std::shared_ptr<GPUOpTuningConf> GPUOpTuningConf::inst_context_ = nullptr;

std::shared_ptr<GPUOpTuningConf> GPUOpTuningConf::GetInstance() {
  static std::once_flag inst_context_init_flag_ = {};
  std::call_once(inst_context_init_flag_, [&]() {
    if (inst_context_ == nullptr) {
      MS_LOG(DEBUG) << "Create new mindspore GPUOpTuningConf";
      inst_context_ = std::make_shared<GPUOpTuningConf>();
    }
  });
  MS_EXCEPTION_IF_NULL(inst_context_);
  return inst_context_;
}

std::string GPUOpTuningConf::conv_fprop_algo() const {
  if (!conv_fprop_algo_.empty()) {
    return conv_fprop_algo_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto conv_fprop_algo = ms_context->get_param<std::string>(MS_CTX_CONV_FPROP_ALGO);
  return conv_fprop_algo;
}

std::string GPUOpTuningConf::conv_wgrad_algo() const {
  if (!conv_wgrad_algo_.empty()) {
    return conv_wgrad_algo_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto conv_wgrad_algo = ms_context->get_param<std::string>(MS_CTX_CONV_WGRAD_ALGO);
  return conv_wgrad_algo;
}

std::string GPUOpTuningConf::conv_dgrad_algo() const {
  if (!conv_dgrad_algo_.empty()) {
    return conv_dgrad_algo_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto conv_dgrad_algo = ms_context->get_param<std::string>(MS_CTX_CONV_DGRAD_ALGO);
  return conv_dgrad_algo;
}

void RegGPUOpTuningConf(py::module *m) {
  (void)py::class_<GPUOpTuningConf, std::shared_ptr<GPUOpTuningConf>>(*m, "GPUOpTuningConf")
    .def_static("get_instance", &GPUOpTuningConf::GetInstance, "Get GPUOpTuningConf instance.")
    .def("set_conv_fprop_algo", &GPUOpTuningConf::set_conv_fprop_algo, "Set Conv Fprop Algo.")
    .def("set_conv_wgrad_algo", &GPUOpTuningConf::set_conv_wgrad_algo, "Set Conv Wgrad Algo.")
    .def("set_conv_dgrad_algo", &GPUOpTuningConf::set_conv_dgrad_algo, "Set Conv Dgrad Algo.")
    .def("is_conv_fprop_algo_configured", &GPUOpTuningConf::IsConvFpropAlgoConfigured, "Set Conv Fprop Algo.")
    .def("is_conv_wgrad_algo_configured", &GPUOpTuningConf::IsConvWgradAlgoConfigured, "Set Conv Wgrad Algo.")
    .def("is_conv_dgrad_algo_configured", &GPUOpTuningConf::IsConvDgradAlgoConfigured, "Set Conv Dgrad Algo.");
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
