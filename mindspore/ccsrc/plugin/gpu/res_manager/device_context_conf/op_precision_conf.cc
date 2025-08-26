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

#include "plugin/gpu/res_manager/device_context_conf/op_precision_conf.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace device {
namespace gpu {
std::shared_ptr<GPUOpPrecisionConf> GPUOpPrecisionConf::inst_context_ = nullptr;

std::shared_ptr<GPUOpPrecisionConf> GPUOpPrecisionConf::GetInstance() {
  static std::once_flag inst_context_init_flag_ = {};
  std::call_once(inst_context_init_flag_, [&]() {
    if (inst_context_ == nullptr) {
      MS_LOG(DEBUG) << "Create new mindspore OpPrecisionConf";
      inst_context_ = std::make_shared<GPUOpPrecisionConf>();
    }
  });
  MS_EXCEPTION_IF_NULL(inst_context_);
  return inst_context_;
}

bool GPUOpPrecisionConf::matmul_allow_tf32() {
  if (IsMatmulAllowTf32Configured()) {
    return matmul_allow_tf32_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto allow_matmul_tf32 = ms_context->get_param<bool>(MS_CTX_MATMUL_ALLOW_TF32);
  return allow_matmul_tf32;
}

bool GPUOpPrecisionConf::conv_allow_tf32() {
  if (IsConvAllowTf32Configured()) {
    return conv_allow_tf32_;
  }
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto conv_allow_tf32 = ms_context->get_param<bool>(MS_CTX_CONV_ALLOW_TF32);
  return conv_allow_tf32;
}

void RegGPUOpPrecisionConf(py::module *m) {
  (void)py::class_<GPUOpPrecisionConf, std::shared_ptr<GPUOpPrecisionConf>>(*m, "GPUOpPrecisionConf")
    .def_static("get_instance", &GPUOpPrecisionConf::GetInstance, "Get GPUOpPrecisionConf instance.")
    .def("matmul_allow_tf32", &GPUOpPrecisionConf::set_matmul_allow_tf32, "Matmul Op allow fp32 to tf32.")
    .def("conv_allow_tf32", &GPUOpPrecisionConf::set_conv_allow_tf32, "Conv Op allow fp32 to tf32.")
    .def("is_matmul_allow_tf32_configured", &GPUOpPrecisionConf::IsMatmulAllowTf32Configured,
         "Matmul Op allow fp32 to tf32")
    .def("is_conv_allow_tf32_configured", &GPUOpPrecisionConf::IsConvAllowTf32Configured, "Conv Op allow fp32 to tf32");
}
}  // namespace gpu
}  // namespace device
}  // namespace mindspore
