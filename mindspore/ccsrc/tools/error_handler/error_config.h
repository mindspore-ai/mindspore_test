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

#ifndef MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_CONFIG_H_
#define MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_CONFIG_H_
#include <map>
#include <string>
#include "include/backend/visible.h"

namespace mindspore {
namespace tools {
class BACKEND_COMMON_EXPORT TftConfig {
 public:
  static bool IsEnableTRE();
  static bool IsEnableStepTRE();
  static int GetSnapShotSteps();

 private:
  static std::map<std::string, std::string> &GetConfigMap();
};
}  // namespace tools
}  // namespace mindspore
#endif  // MINDSPORE_TOOLS_ERROR_HANDLER_ERROR_CONFIG_H_
