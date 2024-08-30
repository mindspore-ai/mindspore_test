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

#include "utils/llm_manager.h"

#include <algorithm>
#include <sstream>
#include "utils/ms_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
LLMManager &LLMManager::GetInstance() noexcept {
  static LLMManager instance;
  return instance;
}

tensor::TensorDataPtr LLMManager::get_graph_input(const std::string &name) {
  auto it = graph_inputs_map_.find(name);
  if (it == graph_inputs_map_.end()) {
    return nullptr;
  }
  return it->second;
}

void LLMManager::add_graph_input(const std::string &name, tensor::TensorDataPtr tensor) {
  graph_inputs_map_[name] = tensor;
}

void LLMManager::reset_graph_inputs() { graph_inputs_map_.clear(); }

void LLMManager::add_force_resize_kernel(const std::string &kernel_name) {
  force_resize_kernel_set_.insert(kernel_name);
  force_resize_kernel_ = true;
}

bool LLMManager::need_force_resize(const std::string &kernel_name) {
  if (!force_resize_kernel_) {
    return false;
  }
  auto it = std::find(force_resize_kernel_set_.begin(), force_resize_kernel_set_.end(), kernel_name);
  return it != force_resize_kernel_set_.end();
}
}  // namespace mindspore
