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

#include "pipeline/llm_boost/llm_boost_binder.h"
#include <iostream>
#include <map>
#include <string>
#include <tuple>
#include <vector>
#include <algorithm>
#include "include/common/utils/convert_utils_py.h"
#include "common/ms_factory.h"
#include "include/common/utils/tensor_py.h"

namespace mindspore {
namespace pipeline {
LlmBoostBinder::LlmBoostBinder(const std::string &boost_type, const std::string &model_name) {
  model_name_ = model_name;
  MS_LOG(INFO) << "Create TransformerBoost, boost type: " << boost_type << ", model_name: " << model_name;
  builder_ = kernel::Factory<kernel::BoostBaseBuilder>::Instance().Create(boost_type);
  if (builder_ == nullptr) {
    MS_LOG(EXCEPTION) << "Get Boost Builder For " << boost_type << " failed";
  }
}

int64_t LlmBoostBinder::Init(const std::string &param) {
  impl_ = builder_->BuildModel(model_name_);
  if (impl_ == nullptr) {
    MS_LOG(EXCEPTION) << "Build Boost Model Failed";
  }
  return impl_->Init(param);
}

std::vector<tensor::TensorPyPtr> LlmBoostBinder::Forward(const py::list &py_inputs, const std::string &param) {
  MS_LOG(INFO) << "TransformerBoost forward";
  std::vector<tensor::TensorPtr> inputs;
  for (auto &obj : py_inputs) {
    auto tensor = IsStubTensor(obj) ? ConvertStubTensor(obj) : tensor::ConvertToTensor(obj);
    inputs.emplace_back(tensor);
  }
  std::vector<tensor::TensorPtr> outputs = impl_->Forward(inputs, param);
  std::vector<tensor::TensorPyPtr> py_outputs;
  (void)std::transform(outputs.begin(), outputs.end(), std::back_inserter(py_outputs),
                       [](const tensor::TensorPtr &p) { return std::make_shared<tensor::TensorPy>(p); });
  return py_outputs;
}

int64_t LlmBoostBinder::SetKVCache(const py::list &py_kcache, const py::list &py_vcache) {
  MS_LOG(INFO) << "TransformerBoost set kv_cache";
  std::vector<tensor::TensorPtr> msKCacheTensors;
  std::vector<tensor::TensorPtr> msVCacheTensors;
  for (auto &obj : py_kcache) {
    auto tensor = IsStubTensor(obj) ? ConvertStubTensor(obj) : tensor::ConvertToTensor(obj);
    msKCacheTensors.emplace_back(tensor);
  }
  for (auto &obj : py_vcache) {
    auto tensor = IsStubTensor(obj) ? ConvertStubTensor(obj) : tensor::ConvertToTensor(obj);
    msVCacheTensors.emplace_back(tensor);
  }
  return impl_->SetKVCache(msKCacheTensors, msVCacheTensors);
}

int64_t LlmBoostBinder::SetWeight(const py::list &py_weights) {
  MS_LOG(INFO) << "LlmBoostBinder::SetWeight";
  std::vector<tensor::TensorPtr> weights;
  for (auto &obj : py_weights) {
    auto tensor = IsStubTensor(obj) ? ConvertStubTensor(obj) : tensor::ConvertToTensor(obj);
    weights.emplace_back(tensor);
  }
  return impl_->SetWeight(weights);
}

int64_t LlmBoostBinder::InitModel(const pybind11::dict &dict) {
  MS_LOG(INFO) << "LlmBoostBinder::InitModel";
  enum dtype_e { VAL_INT, VAL_FLOAT };
  mindspore::kernel::llm_data data;
  std::map<std::string, std::tuple<dtype_e, void *>> keys = {
    {"batch_size", {VAL_INT, &data.batch_size}},   {"seq_length", {VAL_INT, &data.seq_length}},
    {"hidden_size", {VAL_INT, &data.hidden_size}}, {"num_layers", {VAL_INT, &data.num_layers}},
    {"num_heads", {VAL_INT, &data.num_heads}},     {"vocab_size", {VAL_INT, &data.vocab_size}},
    {"multiple_of", {VAL_INT, &data.multiple_of}}, {"rms_norm_eps", {VAL_FLOAT, &data.rms_norm_eps}},
    {"n_kv_heads", {VAL_INT, &data.kv_head_num}},  {"num_blocks", {VAL_INT, &data.page_num}},
    {"block_size", {VAL_INT, &data.page_size}}};

  for (auto &item : dict) {
    auto str = std::string(pybind11::str(item.first));
    auto tup = keys.find(str);
    if (tup != keys.end()) {
      auto [dtype, value] = tup->second;
      if (!item.second.is_none()) {
        switch (dtype) {
          case VAL_INT: {
            *static_cast<int *>(value) = item.second.cast<int>();
            break;
          }
          case VAL_FLOAT: {
            *static_cast<float *>(value) = item.second.cast<float>();
            break;
          }
        }
      }
    }
  }
  if (data.kv_head_num == 0) {
    data.kv_head_num = data.num_heads;
  }
  impl_ = builder_->BuildModel(model_name_);
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "LlmBoostBinder::InitModel: boost is not initialized properly";
    return -1;
  }
  return impl_->InitData(data);
}

int64_t LlmBoostBinder::SetWeightMap(const pybind11::dict &dict) {
  std::map<std::string, mindspore::tensor::TensorPtr> weight_map;
  for (auto &item : dict) {
    auto str = std::string(pybind11::str(item.first));
    auto obj = item.second;
    auto tensor = tensor::ConvertToTensor(obj);
    weight_map[str] = tensor;
  }
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "LlmBoostBinder::SetWeightMap: boost is not initialized properly";
    return -1;
  }

  return impl_->SetWeightMap(weight_map);
}

int64_t LlmBoostBinder::AddFlags(const bool &is_first_iteration) {
  if (impl_ == nullptr) {
    MS_LOG(ERROR) << "LlmBoostBinder::AddFlags: boost is not initialized properly";
    return -1;
  }
  return impl_->AddFlags(is_first_iteration);
}

}  // namespace pipeline
}  // namespace mindspore
