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
#include <utility>
#include "include/runtime/hardware_abstract/device_context/device_context_manager.h"
#include "utils/ms_context.h"
#include "include/utils/pybind_api/api_register.h"
#include "include/runtime/hardware_abstract/stream/multi_stream_controller.h"

namespace mindspore {
namespace {
int CleanTdtChannel() {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_name);
  MS_EXCEPTION_IF_NULL(device_ctx);
  device_ctx->Initialize();
  MS_EXCEPTION_IF_NULL(device_ctx);
  MS_EXCEPTION_IF_NULL(device_ctx->GetKernelExecutor());
  device::DeviceContextManager::GetInstance().SyncAllStreams();
  return device_ctx->GetKernelExecutor()->CleanTdtChannel();
}
}  // namespace

void RegCleanTdtChannel(py::module *m) {
  (void)m->def("clean_tdt_channel", &mindspore::CleanTdtChannel, "Clean tdt channel");
}
}  // namespace mindspore
