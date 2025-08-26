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

#include "tools/error_handler/error_handler.h"
#include <string>
#include <utility>

namespace mindspore {
namespace tools {
namespace {
SNAPSHOT_MANAGER_REG(kCPUDevice, SnapshotMgr);
SNAPSHOT_MANAGER_REG(kGPUDevice, SnapshotMgr);
}  // namespace

SnapshotMgrPtr SnapshotMgr::GetInstance(const std::string &device) {
  auto iter = GetInstanceMap().find(device);
  if (iter == GetInstanceMap().end()) {
    MS_LOG(EXCEPTION) << "Can not find SnapshotMgr for device " << device;
  }
  auto snapshot_mgr = iter->second;
  MS_EXCEPTION_IF_NULL(snapshot_mgr);
  return snapshot_mgr;
}

std::map<std::string, SnapshotMgrPtr> &SnapshotMgr::GetInstanceMap() {
  static std::map<std::string, SnapshotMgrPtr> instance_map = {};
  return instance_map;
}

bool SnapshotMgr::Register(const std::string &device, const SnapshotMgrPtr &instance) {
  auto ret = GetInstanceMap().insert(std::pair<std::string, SnapshotMgrPtr>(device, instance));
  if (ret.second) {
    MS_LOG(INFO) << "SnapshotMgr for device " << device << " is registered successfully.";
  } else {
    MS_LOG(WARNING) << "SnapshotMgr for device " << device << " has already been registered.";
  }
  return true;
}

void SnapshotMgr::Clear() { GetInstanceMap().clear(); }
}  // namespace tools
}  // namespace mindspore
