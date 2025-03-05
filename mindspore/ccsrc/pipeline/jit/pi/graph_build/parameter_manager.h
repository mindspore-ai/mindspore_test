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

#ifndef MINDSPORE_PARAMETER_MANAGER_H
#define MINDSPORE_PARAMETER_MANAGER_H

#include <string>
#include <unordered_map>
#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace mindspore::pijit {

/// \brief A singleton that record and look up Parameter Python objects based on the Parameter name.
class ParameterManager {
 public:
  static ParameterManager &GetInstance() {
    static ParameterManager instance;
    return instance;
  }

  ParameterManager(const ParameterManager &) = delete;
  ParameterManager &operator=(const ParameterManager &) = delete;

  void AddParameter(const std::string &name, py::object parameter);
  py::object FindParameter(const std::string &name);
  void Clear() { parameter_map_.clear(); }

 public:
  /// \brief An RAII object that automatically clean up the Parameters recorded in the ParameterManager.
  class ScopedCleaner {
   public:
    ScopedCleaner() = default;
    ~ScopedCleaner() { ParameterManager::GetInstance().Clear(); }

    ScopedCleaner(const ScopedCleaner &) = delete;
    ScopedCleaner &operator=(const ScopedCleaner &) = delete;
  };

 private:
  ParameterManager() = default;
  ~ParameterManager() = default;

  // Parameter name -> Parameter python object
  std::unordered_map<std::string, py::object> parameter_map_{};
};
}  // namespace mindspore::pijit

#endif  // MINDSPORE_PARAMETER_MANAGER_H
