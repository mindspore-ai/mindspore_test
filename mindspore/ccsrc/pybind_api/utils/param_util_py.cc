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
#include <utility>
#include <vector>
#include <algorithm>
#include "ir/anf.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/ms_context.h"
#include "include/common/pybind_api/api_register.h"
#include "include/common/utils/tensor_py.h"
#include "runtime/device/res_manager/multi_stream_controller.h"

namespace mindspore {
namespace {
int SendRecv(const std::vector<py::object> &params, int src_rank, int dst_rank) {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_name);
  MS_EXCEPTION_IF_NULL(device_ctx);
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  device::DeviceContextManager::GetInstance().SyncAllStreams();
  tensor::TensorPtrList params_;
  (void)std::transform(params.begin(), params.end(), std::back_inserter(params_),
                       [](const py::object &p) { return tensor::ConvertToTensor(p); });
  return device_ctx->device_res_manager_->SendRecv(params_, src_rank, dst_rank);
}

int ResetParams(const std::vector<py::object> &params) {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_name);
  MS_EXCEPTION_IF_NULL(device_ctx);
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  device::DeviceContextManager::GetInstance().SyncAllStreams();
  tensor::TensorPtrList params_;
  (void)std::transform(params.begin(), params.end(), std::back_inserter(params_),
                       [](const py::object &p) { return tensor::ConvertToTensor(p); });
  return device_ctx->device_res_manager_->ResetParams(params_);
}
}  // namespace

void RegSendRecv(py::module *m) {
  (void)m->def("send_recv", &mindspore::SendRecv, "Send and receive parameters", py::arg("params"), py::arg("src_rank"),
               py::arg("dst_rank"));
}

void RegResetParams(py::module *m) {
  (void)m->def("reset_params", &mindspore::ResetParams, "Reset parameter's value to zero", py::arg("params"));
}
}  // namespace mindspore
