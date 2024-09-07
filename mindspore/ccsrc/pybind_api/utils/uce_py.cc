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

#include "include/common/pybind_api/api_register.h"
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"

namespace mindspore {
using DeviceContext = mindspore::device::DeviceContext;
using DeviceContextPtr = std::shared_ptr<DeviceContext>;
namespace {
DeviceContextPtr GetDeviceCtx() {
  const auto &device_name = MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  auto device_ctx = device::DeviceContextManager::GetInstance().GetDeviceContext(device_name);
  if (device_ctx == nullptr) {
    MS_LOG(EXCEPTION) << "Device context of device " << device_name << " is not created yet.";
  }
  return device_ctx;
}
}  // namespace

std::vector<device::DeviceMemPtr> GetMemUceInfo(int32_t device_id) {
  auto device_ctx = GetDeviceCtx();
  return device_ctx->device_res_manager_->GetMemUceInfo(device_id);
}

std::string GetUceProcessStrategy() {
  auto device_ctx = GetDeviceCtx();
  return device_ctx->device_res_manager_->GetUceProcessStrategy();
}

void UceMemRepair(int32_t device_id) {
  auto device_ctx = GetDeviceCtx();
  device_ctx->device_res_manager_->UceMemRepair(device_id);
}

void StopDevice(int32_t device_id) {
  auto device_ctx = GetDeviceCtx();
  device_ctx->device_res_manager_->StopDevice(device_id);
}

void ThrowUCEError() {
  auto device_ctx = GetDeviceCtx();
  device_ctx->device_res_manager_->ThrowUCEError();
}

void RegUCE(py::module *m) {
  (void)m->def("_stop_device", &mindspore::StopDevice, "Stop the device.");
  (void)m->def("_repair_device", &mindspore::UceMemRepair, "Repair the device.");
  (void)m->def("_get_uce_process_strategy", &mindspore::GetUceProcessStrategy, "Get UCE process strategy.");
  (void)m->def("_get_uce_mem_info", &mindspore::GetMemUceInfo, "Get UCE mem info.");
  (void)m->def("_throw_uce_error", &mindspore::ThrowUCEError, "Throw UCE error.");
}
}  // namespace mindspore
