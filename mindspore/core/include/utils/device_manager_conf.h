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
#ifndef MINDSPORE_CORE_INCLUDE_UTILS_DEVICE_MANAGER_CONF_H_
#define MINDSPORE_CORE_INCLUDE_UTILS_DEVICE_MANAGER_CONF_H_

#include <memory>
#include <string>
#include <map>
#include "mindapi/base/macros.h"
#include "utils/log_adapter.h"

namespace mindspore {
const char kDeterministic[] = "deterministic";
const char kDeviceTargetType[] = "device_target";
enum class DeviceTargetType { kUnknown = 0, kCPU = 1, kAscend = 2, kGPU = 3 };

class MS_CORE_API DeviceManagerConf {
 public:
  DeviceManagerConf() = default;
  ~DeviceManagerConf() = default;
  DeviceManagerConf(const DeviceManagerConf &) = delete;
  DeviceManagerConf &operator=(const DeviceManagerConf &) = delete;
  static std::shared_ptr<DeviceManagerConf> GetInstance();

  void set_device(const std::string &device_target, uint32_t device_id, bool is_default_device_id) {
    SetDeviceType(device_target);
    conf_status_[kDeviceTargetType] = true;
    device_id_ = device_id;
    is_default_device_id_ = is_default_device_id;
  }
  const std::string &GetDeviceTarget() {
    auto it = device_type_to_name_map_.find(device_type_);
    if (it == device_type_to_name_map_.end()) {
      MS_EXCEPTION(RuntimeError) << "Can't get the device target. Current wrong device type: " << device_type_;
    }
    return it->second;
  }
  const uint32_t &device_id() { return device_id_; }
  bool is_default_device_id() { return is_default_device_id_; }
  bool IsDeviceEnable() { return conf_status_.count(kDeviceTargetType); }

  void set_deterministic(bool deterministic) {
    deterministic_ = deterministic ? "ON" : "OFF";
    conf_status_[kDeterministic] = true;
  }
  const std::string &deterministic() { return deterministic_; }
  bool IsDeterministicConfigured() { return conf_status_.count(kDeterministic); }

  const DeviceTargetType device_type() const { return device_type_; }
  void SetDeviceType(const std::string &device_target) {
    if (IsDeviceEnable()) {
      return;
    }
    auto it = device_name_to_type_map_.find(device_target);
    if (it != device_name_to_type_map_.end()) {
      device_type_ = it->second;
    }
  }

 private:
  static std::shared_ptr<DeviceManagerConf> instance_;
  static std::map<std::string, DeviceTargetType> device_name_to_type_map_;
  static std::map<DeviceTargetType, std::string> device_type_to_name_map_;

  DeviceTargetType device_type_{DeviceTargetType::kUnknown};
  uint32_t device_id_{0};
  bool is_default_device_id_{true};

  std::string deterministic_{"OFF"};

  std::map<std::string, bool> conf_status_;
};
}  // namespace mindspore

#endif  // MINDSPORE_CORE_INCLUDE_UTILS_DEVICE_MANAGER_CONF_H_
