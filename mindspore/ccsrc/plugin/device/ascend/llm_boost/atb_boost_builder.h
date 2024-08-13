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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_TRANSFOMER_BOOST_ATB_BOOST_BUILDER_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_TRANSFOMER_BOOST_ATB_BOOST_BUILDER_H_
#include <vector>
#include <string>
#include <memory>
#include "include/backend/visible.h"
#include "mindspore/ccsrc/backend/operator/boost_base_builder.h"
namespace mindspore {
namespace kernel {
typedef std::shared_ptr<BoostBaseModel> (*ModelFunc)(std::string);
class BACKEND_EXPORT AtbBoostBuilder : public BoostBaseBuilder {
 public:
  AtbBoostBuilder();
  ~AtbBoostBuilder();
  std::shared_ptr<BoostBaseModel> BuildModel(const std::string &model_name) override;

 private:
  void Initialize();

  void Finalize();

 private:
  void *lib_ptr_{nullptr};
  std::string atb_boost_lib_name_{""};
  const char *model_func_name_ = "CreateAtbBoostModel";
  ModelFunc model_func_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_TRANSFOMER_BOOST_ATB_BOOST_BUILDER_H_
