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
#ifndef MINDSPORE_CCSRC_BACKEND_OPERATE_BOOST_BASE_BUILDER_H_
#define MINDSPORE_CCSRC_BACKEND_OPERATE_BOOST_BASE_BUILDER_H_
#include <optional>
#include <string>
#include <memory>
#include <vector>
#include "ir/tensor.h"
#include "include/backend/visible.h"
#include "backend/operator/boost_base_model.h"

namespace mindspore {
namespace kernel {
class BACKEND_EXPORT BoostBaseBuilder {
 public:
  explicit BoostBaseBuilder(const std::string &boost_name) : boost_name_(boost_name) {}
  ~BoostBaseBuilder() = default;
  virtual std::shared_ptr<BoostBaseModel> BuildModel(const std::string &model_name) = 0;

 protected:
  std::string boost_name_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPERATE_BOOST_BASE_BUILDER_H_
