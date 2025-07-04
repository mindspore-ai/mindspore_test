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

#include "utils/compile_config.h"
#include <string>
#include <utility>
#include "utils/log_adapter.h"
#include "include/utils/ms_utils.h"

namespace mindspore {
CompileConfigManager &CompileConfigManager::GetInstance() noexcept {
  static CompileConfigManager instance;
  return instance;
}

void CompileConfigManager::CollectCompileConfig() {
  if (collect_finished_) {
    return;
  }
  if (collect_func_ == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Compile config not registered.";
  }
  MS_LOG(DEBUG) << "To collect all compile configs.";
  compile_config_ = collect_func_();

  // collect from MS_JIT
  std::istringstream iss(common::GetEnv("MS_JIT"));
  std::string item;

  while (std::getline(iss, item, ',')) {
    size_t pos = item.find(':');
    if (pos != std::string::npos) {
      std::string key = item.substr(0, pos);
      std::string value = item.substr(pos + 1);
      if (compile_config_.find(key) != compile_config_.end()) {
        compile_config_[key] = value;
        MS_LOG(DEBUG) << "Key: " << key << ", Value: " << value;
      } else {
        MS_LOG(EXCEPTION) << "MS_JIT does not support parameter: " << key;
      }
    } else if (item != "0" && item != "1") {
      MS_LOG(EXCEPTION) << "MS_JIT does not support parameter: " << item;
    }
  }

  collect_finished_ = true;
}

void CompileConfigManager::SetConfig(const std::string &config_name, const std::string &value, bool overwrite) {
  if (!overwrite && compile_config_.find(config_name) != compile_config_.end()) {
    return;
  }
  compile_config_.insert_or_assign(config_name, value);
}

std::string CompileConfigManager::GetConfig(const std::string &config_name) {
  if (compile_config_.empty()) {
    MS_LOG(INFO) << "The compile config is empty when getting config '" << config_name << "'.";
    return "";
  }
  auto iter = compile_config_.find(config_name);
  if (iter == compile_config_.end()) {
    MS_LOG(INTERNAL_EXCEPTION) << "'" << config_name << "' is not a compile config.";
  }
  MS_LOG(DEBUG) << "Get Compile Config. " << config_name << ": " << iter->second;
  return iter->second;
}

namespace common {
std::string GetCompileConfig(const std::string &config_name) {
  return CompileConfigManager::GetInstance().GetConfig(config_name);
}

void SetCompileConfig(const std::string &config_name, const std::string &value, bool overwrite) {
  CompileConfigManager::GetInstance().SetConfig(config_name, value, overwrite);
}
}  // namespace common
}  // namespace mindspore
