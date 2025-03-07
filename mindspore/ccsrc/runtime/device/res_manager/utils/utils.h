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
#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_UTILS_UTILS_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_UTILS_UTILS_H_

#include <string>
#include <vector>
#include "common/device_type.h"
#include "runtime/device/res_manager/utils/visible.h"
#include "ir/tensor.h"

namespace mindspore {
namespace device {
struct ResKey {
  DeviceType device_name_;
  uint32_t device_id_{0};
  std::string ToString() const { return GetDeviceNameByType(device_name_) + "_" + std::to_string(device_id_); }

  std::string DeviceName() const { return GetDeviceNameByType(device_name_); }
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_RES_MANAGER_UTILS_UTILS_H_
