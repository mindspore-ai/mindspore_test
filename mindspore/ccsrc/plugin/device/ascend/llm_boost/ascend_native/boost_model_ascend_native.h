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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_LLM_BOOST_ASCEND_NATIVE_BOOST_MODEL_ASCEND_NATIVE_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_LLM_BOOST_ASCEND_NATIVE_BOOST_MODEL_ASCEND_NATIVE_H_
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include "include/backend/visible.h"
#include "mindspore/ccsrc/backend/operator/boost_base_model.h"
namespace mindspore {
namespace kernel {

class BACKEND_EXPORT BoostModelAscendC : public BoostBaseModel {
 public:
  explicit BoostModelAscendC(const std::string &model_name) : model_name_(model_name) {}
  ~BoostModelAscendC() = default;
  int64_t Init(const std::string &param) override;
  int64_t InitData(const llm_data &data) override;
  std::vector<tensor::TensorPtr> Forward(const std::vector<tensor::TensorPtr> &input,
                                         const std::string &param) override;
  int64_t SetWeight(const std::vector<tensor::TensorPtr> &weights) override { return -1; }
  int64_t SetWeightMap(const std::map<std::string, mindspore::tensor::TensorPtr> &map) override;

  int64_t SetKVCache(const std::vector<tensor::TensorPtr> &msKCacheTensors,
                     const std::vector<tensor::TensorPtr> &msVCacheTensors) override {
    return -1;
  }
  int64_t AddFlags(bool is_first) override;

 private:
  std::string model_name_;
  std::shared_ptr<void> llama_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_LLM_BOOST_ASCEND_NATIVE_BOOST_MODEL_ASCEND_NATIVE_H_
