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
#include "utils/log_adapter.h"
#include "ir/tensor.h"

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

  tensor::TensorPtr get_graph_input(const std::string &name);

  void add_graph_input(const std::string &name, tensor::TensorPtr tensor);

  void reset_graph_inputs();

  void add_force_resize_kernel(const std::string &kernel_name);

  bool need_force_resize(const std::string &kernel_name);

  void Clear();

  template <typename T1, typename T2>
  bool GetGraphInputToVector(const std::string &tensor_name, std::vector<T2> *output) {
    auto in_tensor = get_graph_input(tensor_name);
    if (in_tensor == nullptr) {
      output->clear();
      return false;
    }
    // then use graph_input tensor value to set seq_len if saved
    auto in_tensor_data = static_cast<T1 *>(in_tensor->data_c());
    if (in_tensor_data == nullptr) {
      output->clear();
      return false;
    }
    auto in_tensor_data_num = in_tensor->Size();
    output->resize(in_tensor_data_num);

    for (size_t i = 0; i < in_tensor_data_num; i++) {
      (*output)[i] = static_cast<T2>(in_tensor_data[i]);
    }

    return true;
  }

 private:
  bool force_resize_kernel_{false};
  std::map<std::string, tensor::TensorPtr> graph_inputs_map_;
  std::set<std::string> force_resize_kernel_set_{};
};

}  // namespace mindspore
#endif  // MINDSPORE_CORE_UTILS_LLM_MANAGER_H_
