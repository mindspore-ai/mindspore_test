/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "utils/phase.h"
#include <mutex>

namespace mindspore {
PhaseManager &PhaseManager::GetInstance() noexcept {
  static PhaseManager instance;
  return instance;
}

void PhaseManager::ClearJitConfig() {
  std::unique_lock<std::shared_mutex> lock(rw_jit_mutex_);
  jit_config_.clear();
}

void PhaseManager::set_jit_config(const std::map<std::string, std::string> &jit_config) {
  std::unique_lock<std::shared_mutex> lock(rw_jit_mutex_);
  jit_config_ = jit_config;
}

const std::map<std::string, std::string> &PhaseManager::jit_config() const {
  std::shared_lock<std::shared_mutex> lock(rw_jit_mutex_);
  return jit_config_;
}

std::string PhaseManager::GetJitBackend() const {
  std::shared_lock<std::shared_mutex> lock(rw_jit_mutex_);
  auto iter = jit_config_.find("backend");
  if (iter == jit_config_.end()) {
    return "";
  }
  return iter->second;
}

std::string PhaseManager::GetJitLevel() const {
  std::shared_lock<std::shared_mutex> lock(rw_jit_mutex_);
  auto iter = jit_config_.find("jit_level");
  if (iter == jit_config_.end()) {
    return "";
  }
  return iter->second;
}
}  // namespace mindspore
