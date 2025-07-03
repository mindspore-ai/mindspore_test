/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_GRAPH_ADAPTER_H_
#define MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_GRAPH_ADAPTER_H_

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "ir/kernel_tensor_value.h"
#include "mindspore/core/include/base/base.h"
#include "mindspore/core/include/base/base_ref.h"
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"

#include "dalang/dair/tensor/tensor.h"
#include "dalang/dart/runtime/executor.h"

namespace mindspore {
namespace backend {
namespace ms_infer_backend {

class GraphAdapter {
 public:
  explicit GraphAdapter(const FuncGraphPtr &func_graph) : func_graph_(func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph_);
    device_context_ = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET),
       MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    device_context_->Initialize();
  }
  ~GraphAdapter() {}

  void ConvertGraph();
  void RunGraph(const VectorRef &inputs, VectorRef *outputs);

 private:
  void ConvertParameters();
  void InsertParameters();
  void ConvertCNodes();
  void ConvertCNode(const CNodePtr &node);
  void ConvertInputs(const VectorRef &inputs);
  void ConvertOutputs(VectorRef *outputs);

  void SetValue(da::tensor::DATensor *da_value, const BaseRef &val);
  da::tensor::DATensor *GetNodeDATensor(const AnfNodePtr &node);

  void *PrepareData(da::tensor::DATensor *da_value, const ValuePtr &value);
  void *PrepareTensorDataToDevice(const tensor::TensorPtr &tensor);

  FuncGraphPtr func_graph_;
  da::runtime::GraphExecutor graph_executor_;
  std::unordered_map<AnfNodePtr, da::tensor::DATensor *> apply_map_;
  std::unordered_map<AnfNodePtr, da::tensor::DATensor *> const_map_;
  std::unordered_map<AnfNodePtr, da::tensor::DATensor *> parameter_map_;
  std::unordered_set<ValuePtr> converted_values_;
  device::DeviceContext *device_context_{nullptr};
};

using GraphAdapterPtr = std::shared_ptr<GraphAdapter>;

}  // namespace ms_infer_backend
}  // namespace backend
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_MS_INFER_BACKEND_GRAPH_ADAPTER_H_
