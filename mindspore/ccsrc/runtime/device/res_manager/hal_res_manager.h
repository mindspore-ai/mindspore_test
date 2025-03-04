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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_HAL_RES_MANAGER_H
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_HAL_RES_MANAGER_H
#include <string>
#include <vector>
#include <map>
#include <functional>
#include <utility>
#include <memory>

#include "include/backend/visible.h"
#include "runtime/device/res_manager/hal_res_base.h"
#include "runtime/device/res_manager/multi_stream_controller.h"

namespace mindspore {
namespace device {
using HalResManagerCreator = std::function<std::shared_ptr<HalResBase>(const ResKey &)>;

class RES_EXPORT HalResManager {
 public:
  ~HalResManager() = default;
  static HalResManager &GetInstance();
  void Register(const DeviceType device, HalResManagerCreator &&hal_res_manager_creator);
  HalResBase *GetOrCreateResManager(const ResKey &res_key);
  HalResPtr GetResManager(const ResKey &res_key);

  MultiStreamControllerPtr &GetMultiStreamController(const std::string &device_name);

 private:
  std::map<DeviceType, HalResManagerCreator> hal_res_manager_creators_;
  std::map<std::string, HalResPtr> res_managers_;

  // Since multi device is not supported currently, here use device target type to improve performance.
  // Device target type : 0, 1, 2, 3, and real device support : 'GPU' 'Ascend' 'CPU'.
  std::map<std::string, MultiStreamControllerPtr> multi_stream_controllers_;
};
class RES_EXPORT HalResManagerRegister {
 public:
  HalResManagerRegister(const DeviceType device, HalResManagerCreator &&hal_res_manager_creator) {
    HalResManager::GetInstance().Register(device, std::move(hal_res_manager_creator));
  }
  ~HalResManagerRegister() = default;
};

#define MS_REGISTER_HAL_RES_MANAGER(DEVICE_NAME, TARGET, HAL_RES_MANAGER_CLASS) \
  static const HalResManagerRegister g_res_mananger_##DEVICE_NAME##_reg(        \
    TARGET, [](const ResKey &res_key) { return std::make_shared<HAL_RES_MANAGER_CLASS>(res_key); });

}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_HAL_RES_MANAGER_H
