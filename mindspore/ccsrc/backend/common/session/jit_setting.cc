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
#include "include/backend/jit_setting.h"
#include <vector>
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include "utils/phase.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace session {
JitSetting JitSetting::ParseJitSetting() {
  JitSetting jit_setting;

  const auto &jit_config = PhaseManager::GetInstance().jit_config();
  auto iter = jit_config.find("jit_level");
  if (iter != jit_config.end()) {
    jit_setting.jit_level = iter->second;
  }
  iter = jit_config.find("backend");
  if (iter != jit_config.end()) {
    jit_setting.backend = iter->second;
  }

  iter = jit_config.find("options");
  if (iter != jit_config.end()) {
    nlohmann::json options_json = nlohmann::json::parse(iter->second);
    if (options_json.contains("disable_format_transform")) {
      options_json["disable_format_transform"].get_to(jit_setting.disable_format_transform);
    }
    if (options_json.contains("exec_order")) {
      options_json["exec_order"].get_to(jit_setting.exec_order);
    }
    if (options_json.contains("ge_options")) {
      options_json["ge_options"].get_to(jit_setting.ge_options);
    }
  }

  return jit_setting;
}

nlohmann::json JitSetting::to_json() const {
  auto ret = nlohmann::json{{"jit_level", jit_level},
                            {"backend", backend},
                            {"disable_format_transform", disable_format_transform},
                            {"exec_order", exec_order},
                            {"ge_options", ge_options}};
  return ret;
}

void JitSetting::from_json(const nlohmann::json &j) {
  j.at("jit_level").get_to(jit_level);
  j.at("backend").get_to(backend);
  j.at("disable_format_transform").get_to(disable_format_transform);
  j.at("exec_order").get_to(exec_order);
  j.at("ge_options").get_to(ge_options);
}
}  // namespace session
}  // namespace mindspore
