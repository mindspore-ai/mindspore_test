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
#include "include/common/utils/convert_utils_py.h"
#include "plugin/factory/ms_factory.h"

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

std::vector<tensor::TensorPtr> LlmBoostBinder::Forward(const py::list &py_inputs, const std::string &param) {
  MS_LOG(INFO) << "TransformerBoost forward";
  std::vector<tensor::TensorPtr> inputs;
  for (auto &obj : py_inputs) {
    auto tensor = IsStubTensor(obj) ? ConvertStubTensor(obj) : obj.cast<tensor::TensorPtr>();
    inputs.emplace_back(tensor);
  }
  return impl_->Forward(inputs, param);
}

int64_t LlmBoostBinder::SetKVCache(const py::list &py_kcache, const py::list &py_vcache) {
  MS_LOG(INFO) << "TransformerBoost set kv_cache";
  std::vector<tensor::TensorPtr> msKCacheTensors;
  std::vector<tensor::TensorPtr> msVCacheTensors;
  for (auto &obj : py_kcache) {
    auto tensor = IsStubTensor(obj) ? ConvertStubTensor(obj) : obj.cast<tensor::TensorPtr>();
    msKCacheTensors.emplace_back(tensor);
  }
  for (auto &obj : py_vcache) {
    auto tensor = IsStubTensor(obj) ? ConvertStubTensor(obj) : obj.cast<tensor::TensorPtr>();
    msVCacheTensors.emplace_back(tensor);
  }
  return impl_->SetKVCache(msKCacheTensors, msVCacheTensors);
}

int64_t LlmBoostBinder::SetWeight(const py::list &py_weights) {
  MS_LOG(INFO) << "TransformerBoost set_weights";
  std::vector<tensor::TensorPtr> weights;
  for (auto &obj : py_weights) {
    auto tensor = IsStubTensor(obj) ? ConvertStubTensor(obj) : obj.cast<tensor::TensorPtr>();
    weights.emplace_back(tensor);
  }
  return impl_->SetWeight(weights);
}
}  // namespace pipeline
}  // namespace mindspore
