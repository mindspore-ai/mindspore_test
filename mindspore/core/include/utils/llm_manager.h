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

#ifndef MINDSPORE_CORE_UTILS_LLM_MANAGER_H_
#define MINDSPORE_CORE_UTILS_LLM_MANAGER_H_

#include <string>
#include <memory>
#include <map>
#include <vector>
#include <set>
#include "mindapi/base/macros.h"
#include "ir/meta_tensor.h"
#include "utils/log_adapter.h"
#include "ir/tensor_data.h"

namespace mindspore {
// Current not support multi -thread use this Single Instance
class MS_CORE_API LLMManager {
 public:
  /// \brief Get instance of LLMManager.
  ///
  /// \return Instance of LLMManager.
  static LLMManager &GetInstance() noexcept;

  /// \brief Disable the default copy constructor.
  LLMManager &operator=(const LLMManager &) = delete;
  /// \brief Destructor.
  ~LLMManager() = default;

  tensor::TensorDataPtr get_graph_input(const std::string &name);

  void add_graph_input(const std::string &name, tensor::TensorDataPtr tensor);

  void reset_graph_inputs();

  void add_force_resize_kernel(const std::string &kernel_name);

  bool need_force_resize(const std::string &kernel_name);

  void Clear();

 private:
  bool force_resize_kernel_{false};
  std::map<std::string, tensor::TensorDataPtr> graph_inputs_map_;
  std::set<std::string> force_resize_kernel_set_{};
};
}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_LLM_MANAGER_H_
