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
#include "pybind_api/utils/stress_detect_py.h"
#include <utility>
#include "runtime/pynative/op_executor.h"
#include "runtime/pipeline/pipeline.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/ms_context.h"
#include "include/common/pybind_api/api_register.h"
#include "runtime/device/res_manager/hal_res_manager.h"

namespace mindspore {
namespace {
DeviceContext *GetDeviceCtx() {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
    {device_name, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
  MS_EXCEPTION_IF_NULL(device_ctx);

  device_ctx->Initialize();
  return device_ctx;
}
}  // namespace

int StressDetect() {
  auto device_ctx = GetDeviceCtx();
  MS_EXCEPTION_IF_NULL(device_ctx);
  runtime::Pipeline::Get().WaitAll();
  auto &controller =
    device::HalResManager::GetInstance().GetMultiStreamController(device_ctx->device_context_key().device_name_);
  controller->Refresh();
  (void)controller->SyncAllStreams();
  MS_EXCEPTION_IF_NULL(device_ctx->device_res_manager_);
  return device_ctx->device_res_manager_->StressDetect();
}

void RegStress(py::module *m) { (void)m->def("stress_detect", &mindspore::StressDetect, "Detect stress"); }
}  // namespace mindspore
