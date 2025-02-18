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
#ifndef MINDSPORE_CCSRC_BACKEND_SESSION_JIT_SETTING_H
#define MINDSPORE_CCSRC_BACKEND_SESSION_JIT_SETTING_H

#include <vector>
#include <string>
#include <map>
#include <nlohmann/json.hpp>
#include "ir/graph_utils.h"
#include "include/common/utils/utils.h"
#include "include/backend/visible.h"

namespace mindspore {
namespace session {
struct BACKEND_EXPORT JitSetting {
  // parse jit setting from PhaseManager::GetInstance().jit_config()
  static JitSetting ParseJitSetting();

  nlohmann::json to_json() const;
  void from_json(const nlohmann::json &j);

  // jit level, O0/O1
  std::string jit_level = "";
  // backend, ms_backend/GE, may not be set when using graphpipeline
  std::string backend = "";
  // Whether to disable the automatic format transform function from NCHW to NHWC
  bool disable_format_transform = false;
  // Sorting method for operator execution in KBK
  std::string exec_order = "";
  // GE options, {"graph":{option:value}, "session":{option:value}, "graph":{option:value}}
  std::map<std::string, std::map<std::string, std::string> > ge_options = {};
};
}  // namespace session
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_SESSION_JIT_SETTING_H
