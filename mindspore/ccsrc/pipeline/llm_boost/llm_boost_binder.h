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

#ifndef MINDSPORE_CCSRC_PIPELINE_LLM_BOOST_LLM_BOOST_BINDER_H_
#define MINDSPORE_CCSRC_PIPELINE_LLM_BOOST_LLM_BOOST_BINDER_H_

#include <vector>
#include <string>
#include <memory>
#include "pybind11/pybind11.h"
#include "include/common/visible.h"
#include "backend/operator/boost_base_builder.h"
#include "include/common/utils/python_adapter.h"
#include "include/common/utils/tensor_py.h"

namespace mindspore {
namespace pipeline {
class ME_EXPORT LlmBoostBinder {
 public:
  explicit LlmBoostBinder(const std::string &backend, const std::string &model_name);
  ~LlmBoostBinder() = default;

  int64_t Init(const std::string &param);
  std::vector<py::object> Forward(const py::list &py_inputs, const std::string &param);
  int64_t SetKVCache(const py::list &py_kcache, const py::list &py_vcache);
  int64_t SetWeight(const py::list &py_weights);
  int64_t AddFlags(const bool &is_first_iteration);
  int64_t SetWeightMap(const pybind11::dict &dict);
  int64_t InitModel(const pybind11::dict &dict);

 private:
  std::shared_ptr<kernel::BoostBaseModel> impl_;
  std::shared_ptr<kernel::BoostBaseBuilder> builder_;
  std::string model_name_;
  std::string backend_;
};
}  // namespace pipeline
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PIPELINE_LLM_BOOST_LLM_BOOST_BINDER_H_
