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

#include <cstring>
#include <unordered_set>
#include <string>
#include <vector>
#include <algorithm>
#include <iterator>

const std::unordered_set<std::string> register_op_name = {"LogicalAnd", "CumsumExt", "InplaceReLU",
                                                          "Randn",      "StackExt",  "SumExt"};

extern "C" {

bool IsKernelRegistered(const char *name) {
  if (name == nullptr) {
    return false;
  }
  return register_op_name.find(name) != register_op_name.end();
}

int GetRegisteredOpCount() { return register_op_name.size(); }

const char **GetAllRegisteredOps() {
  static std::vector<const char *> op_names;
  static bool initialized = false;
  if (!initialized) {
    op_names.reserve(register_op_name.size());
    std::transform(register_op_name.begin(), register_op_name.end(), std::back_inserter(op_names),
                   [](const std::string &name) { return name.c_str(); });
    initialized = true;
  }
  return op_names.data();
}
}  // extern "C"
