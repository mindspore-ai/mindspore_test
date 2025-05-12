/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "runtime/device/res_manager/hal_res_manager.h"

#include <memory>

#include "utils/ms_context.h"

namespace mindspore {
namespace device {
HalResManager &HalResManager::GetInstance() {
  static HalResManager instance{};
  return instance;
}

void HalResManager::Register(const DeviceType device, HalResManagerCreator &&hal_res_manager_creator) {
  if (hal_res_manager_creators_.find(device) == hal_res_manager_creators_.end()) {
    (void)hal_res_manager_creators_.emplace(device, hal_res_manager_creator);
  }
}

HalResBase *HalResManager::GetOrCreateResManager(const ResKey &res_key) {
  auto res_manager_iter = res_managers_.find(res_key.ToString());
  if (res_manager_iter != res_managers_.end()) {
    return res_manager_iter->second.get();
  }

  std::shared_ptr<HalResBase> res_manager;
  auto creator_iter = hal_res_manager_creators_.find(res_key.device_name_);
  if (creator_iter != hal_res_manager_creators_.end()) {
    res_manager = (creator_iter->second)(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);
    res_managers_[res_key.ToString()] = res_manager;
    multi_stream_controllers_[res_key.DeviceName()] = std::make_shared<MultiStreamController>(res_manager.get());
  } else {
    MS_LOG(EXCEPTION) << "Create resource manager failed, please make sure target device:" << res_key.ToString()
                      << " is valid.";
  }
  return res_manager.get();
}
HalResPtr HalResManager::GetResManager(const ResKey &res_key) {
  if (res_managers_.count(res_key.ToString()) == 0) {
    MS_LOG(INFO) << "ResManager of device " << res_key.ToString() << " is not created yet.";
    return nullptr;
  }
  return res_managers_[res_key.ToString()];
}

MultiStreamControllerPtr &HalResManager::GetMultiStreamController(const std::string &device_name) {
  auto &&iter = multi_stream_controllers_.find(device_name);
  if (iter != multi_stream_controllers_.end()) {
    return iter->second;
  }
  MS_LOG(WARNING) << "Found multi stream controller failed, and try to initialize, device_name : " << device_name
                  << ".";
  auto device_id = MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto res_key = ResKey{GetDeviceTypeByName(device_name), device_id};
  auto hal_res_base = GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(hal_res_base);
  auto &&iter_again = multi_stream_controllers_.find(device_name);
  if (iter_again == multi_stream_controllers_.end()) {
    MS_LOG(EXCEPTION) << "Get multi stream controller failed, device_name : " << device_name << ".";
  }
  return iter_again->second;
}
}  // namespace device
}  // namespace mindspore
