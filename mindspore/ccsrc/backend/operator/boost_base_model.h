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
#ifndef MINDSPORE_CCSRC_BACKEND_OPERATE_BOOST_BASE_MODEL_H_
#define MINDSPORE_CCSRC_BACKEND_OPERATE_BOOST_BASE_MODEL_H_
#include <optional>
#include <vector>
#include <string>
#include "ir/tensor.h"
#include "include/backend/visible.h"
namespace mindspore {
namespace kernel {
class BACKEND_EXPORT BoostBaseModel {
 public:
  BoostBaseModel() {}
  ~BoostBaseModel() = default;
  virtual int64_t Init(const std::string &param) = 0;
  virtual std::vector<tensor::TensorPtr> Forward(const std::vector<tensor::TensorPtr> &input,
                                                 const std::string &param) = 0;
  virtual int64_t SetWeight(const std::vector<tensor::TensorPtr> &weights) = 0;
  virtual int64_t SetKVCache(const std::vector<tensor::TensorPtr> &msKCacheTensors,
                             const std::vector<tensor::TensorPtr> &msVCacheTensors) = 0;
  std::string modelName_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPERATE_BOOST_BASE_MODEL_H_
