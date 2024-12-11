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
#ifndef MINDSPORE_CCSRC_BACKEND_OPERATOR_BOOST_BASE_MODEL_H_
#define MINDSPORE_CCSRC_BACKEND_OPERATOR_BOOST_BASE_MODEL_H_
#include <map>
#include <optional>
#include <string>
#include <vector>
#include "ir/tensor.h"
#include "include/backend/visible.h"
namespace mindspore {
namespace kernel {

struct llm_data {
  int batch_size;
  int seq_length;
  int hidden_size;
  int num_layers;
  int num_heads;
  int vocab_size;
  int multiple_of;
  int kv_head_num = 0;
  int page_num = 0;
  int page_size = 0;
  float rms_norm_eps;
};

class BACKEND_EXPORT BoostBaseModel {
 public:
  BoostBaseModel() {}
  ~BoostBaseModel() = default;
  virtual int64_t Init(const std::string &param) = 0;
  virtual int64_t InitData(const llm_data &data) = 0;
  virtual std::vector<tensor::TensorPtr> Forward(const std::vector<tensor::TensorPtr> &input,
                                                 const std::string &param) = 0;
  virtual int64_t SetWeight(const std::vector<tensor::TensorPtr> &weights) = 0;
  virtual int64_t SetWeightMap(const std::map<std::string, mindspore::tensor::TensorPtr> &map) = 0;

  virtual int64_t SetKVCache(const std::vector<tensor::TensorPtr> &msKCacheTensors,
                             const std::vector<tensor::TensorPtr> &msVCacheTensors) = 0;
  virtual int64_t AddFlags(bool is_first) = 0;

  std::string modelName_;
};
#define MS_BOOST_MODEL_FACTORY_REG(NAME, DERIVE) MS_KERNEL_FACTORY_REG(BoostBaseModel, NAME, DERIVE)
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPERATOR_BOOST_BASE_MODEL_H_
