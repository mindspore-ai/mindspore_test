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

#include "frontend/jit/pi/graph_build/parameter_manager.h"

#include <string>
#include <unordered_map>
#include <utility>
#include "pybind11/pybind11.h"

#include "utils/log_adapter.h"

namespace mindspore::pijit {

void ParameterManager::AddParameter(const std::string &name, py::object parameter) {
  if (name.empty()) {
    MS_LOG(DEBUG) << "Parameter name is empty string, skip it!";
    return;
  }
  if (parameter.ptr() == nullptr) {
    MS_LOG(DEBUG) << "Parameter obj is null, skip it: " << name;
    return;
  }
  MS_LOG(DEBUG) << "Add parameter: " << name;
  parameter_map_[name] = std::move(parameter);
}

py::object ParameterManager::FindParameter(const std::string &name) {
  auto it = parameter_map_.find(name);
  return it != parameter_map_.end() ? it->second : py::object();
}
}  // namespace mindspore::pijit
