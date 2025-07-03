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
#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_LLM_BOOST_ATB_BOOST_MODEL_ATB_H_
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_LLM_BOOST_ATB_BOOST_MODEL_ATB_H_
#include <vector>
#include <string>
#include <memory>
#include <map>
#include "include/backend/visible.h"
#include "mindspore/ccsrc/backend/operator/boost_base_model.h"
#include "runtime/hardware/device_context.h"
#include "plugin/device/ascend/llm_boost/atb/workspace.h"

#include "atb_speed/base/model.h"
#include "atb_speed/utils/timer.h"

namespace mindspore {
namespace kernel {
class BACKEND_EXPORT BoostModelATB : public BoostBaseModel {
 public:
  explicit BoostModelATB(const std::string modelName);
  ~BoostModelATB();
  int64_t Init(const std::string &param) override;
  std::vector<tensor::TensorPtr> Forward(const std::vector<tensor::TensorPtr> &input,
                                         const std::string &param) override;
  void InitContext();

  void *GetWorkSpace(uint64_t bufferSize, uint32_t bufferKey);
  atb::Tensor CreateInternalTensorFromDesc(const atb::TensorDesc &tensorDesc);
  atb::Tensor MSTensor2Tensor(const tensor::TensorPtr &msTensor);
  int64_t MSTensor2Tensor(const std::vector<tensor::TensorPtr> &msTensors, std::vector<atb::Tensor> &opsTensors);
  const tensor::TensorPtr CreateMsTensorFromTensorDesc(const atb::TensorDesc &tensorDesc);

  int64_t SetWeight(const std::vector<tensor::TensorPtr> &weights) override;
  int64_t SetKVCache(const std::vector<tensor::TensorPtr> &msKCacheTensors,
                     const std::vector<tensor::TensorPtr> &msVCacheTensors) override;
  int64_t ExecuteOutImpl(std::vector<atb::Tensor> &inTensors, std::vector<atb::Tensor> &outTensors,
                         const std::string &param);
  int64_t SetWeightMap(const std::map<std::string, mindspore::tensor::TensorPtr> &map) override { return -1; }
  int64_t InitData(const llm_data &data) override { return -1; }
  int64_t AddFlags(bool is_first) override { return -1; }

 private:
  void InitKVCacheTensor();
  void RunTask(std::string taskName, std::function<int()> task);

 private:
  std::string modelName_{""};
  std::shared_ptr<atb_speed::Model> model_{nullptr};
  uint64_t executeCount_{0};
  uint64_t modelId_{0};
  std::shared_ptr<atb::Context> context_{nullptr};
  std::vector<tensor::TensorPtr> msInternalTensors_;
  uint64_t stream_id_{0};
  device::DeviceContext *device_context_{nullptr};
  std::string device_name_{""};
  uint64_t device_id_{0};
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_ASCEND_LLM_BOOST_ATB_BOOST_MODEL_ATB_H_
