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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_DATA_PREPARE_ACTOR_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_DATA_PREPARE_ACTOR_H_

#include <atomic>
#include <vector>
#include <string>
#include <memory>
#include <utility>
#include <map>
#include <set>

#include "utils/hash_map.h"
#include "backend/ge_backend/runtime/graph_compiler.h"
#include "backend/ge_backend/runtime/actor/actor_common.h"
#include "backend/ge_backend/runtime/actor/data_source_actor.h"
#include "backend/ge_backend/runtime/actor/debug_aware_actor.h"
#include "backend/ge_backend/runtime/device_tensor_store.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {

// The data prepare actor is used to prepare data for device tensor store and host tensor queue to represent the begin
// of one step.
class DataPrepareActor : public DebugAwareActor {
 public:
  DataPrepareActor(const std::string &name, const AID &memory_manager_aid, const AID *debug_aid,
                   const AID *profiler_aid, const GraphCompilerInfo *graph_compiler_info,
                   const HostQueueDSActorPtr &host_data_source_actor, const HostTensorQueuePtr &host_tensor_queue)
      : DebugAwareActor(name, KernelTransformType::kDataPrepareActor, nullptr, memory_manager_aid, debug_aid,
                        profiler_aid),
        graph_compiler_info_(graph_compiler_info),
        strategy_(GraphExecutionStrategy::kPipeline),
        real_strategy_(GraphExecutionStrategy::kPipeline),
        host_data_source_actor_(host_data_source_actor),
        host_tensor_queue_(host_tensor_queue),
        first_step_(true),
        has_parameter_input_(false) {}
  ~DataPrepareActor() override = default;

  // The process entry of data prepare.
  void PrepareData(const std::vector<std::vector<TensorPtr>> &input_tensors, const VectorRef &args,
                   OpContext<KernelTensor> *const context, GraphExecutionStrategy real_strategy);

  // The debug related operation interface.
  void SendDebugReq(OpContext<KernelTensor> *const context) override;
  void SendProfilerReq(OpContext<KernelTensor> *const context);
  void OnDebugFinish(OpContext<KernelTensor> *const context) override;

 protected:
  void Init() override;
  void Run(OpContext<KernelTensor> *const context) override {
    VectorRef empty_args;
    PrepareData(init_tensors_, empty_args, context, GraphExecutionStrategy::kPipeline);
  }

 private:
  friend class GraphScheduler;

  void UpdateDynamicShapeAndSize(const AnfNodePtr &input_node, const TensorPtr &input_tensor) const;
  void UpdateDeviceAddressForDataNode(const AnfNodePtr &input_node, const TensorPtr &input_tensor);

  // Fetch the input info.
  TensorPtr FetchInputTensor(const std::vector<TensorPtr> &tensors, size_t tensor_index, const VectorRef &args,
                             const KernelWithIndex &front_node);

  void PrepareDataForDeviceTensorStore(const std::vector<std::vector<TensorPtr>> &input_tensors, const VectorRef &args,
                                       OpContext<KernelTensor> *const context);
  void PrepareDataForHostTensorQueue(const std::vector<std::vector<TensorPtr>> &input_tensors, const VectorRef &args,
                                     OpContext<KernelTensor> *const context);
  void PrepareDataForHostTensorQueueNew(const VectorRef &args, OpContext<KernelTensor> *const context);

  // Prepare the device data for persistent device tensor of weight node from host tensor.
  void PrepareDataForWeightNode(const AnfNodePtr &backend_node, const AnfNodePtr &front_node, const TensorPtr &tensor,
                                OpContext<KernelTensor> *const context);
  // Prepare the device data for persistent device tensor of value node.
  void PrepareDataForValueNode(const ValueNodePtr &node, const AnfNodePtr &front_node,
                               OpContext<KernelTensor> *const context) const;
  void PrepareDataForStringValue(const ValueNodePtr &node, size_t index, const AnfNodePtr &front_node,
                                 OpContext<KernelTensor> *const context) const;
  // Sync host data of Sequence or Scalar type value to device side.
  void PrepareDataForSequenceAndScalarValue(const ValueNodePtr &node, size_t index, const AnfNodePtr &front_node,
                                            OpContext<KernelTensor> *const context) const;
  //  The branch processing of PrepareDataForValueNode that value type is tensor.
  void PrepareDataForValueNodeTensor(const ValueNodePtr &node, const ValuePtr &node_value, const AnfNodePtr &front_node,
                                     OpContext<KernelTensor> *const context) const;

  // The data prepare in the control flow scene.
  // If the parameters in the root graph are only used by the control node, these parameters will not be initialized
  // by the kernel graph, and addresses need to be specially allocated for these parameters.
  void PrepareDeviceTensorStoreForControlNode(const ControlNodeParserPtr &control_node_parser,
                                              const std::vector<TensorPtr> &tensors, const VectorRef &args,
                                              OpContext<KernelTensor> *const context);
  void PrepareHostTensorQueueForControlNode(const std::vector<TensorPtr> &tensors,
                                            std::vector<TensorPtr> *const host_tensors,
                                            OpContext<KernelTensor> *const context);
  void PrepareDataForControlValueNode(const KernelWithIndex &node_with_index, OpContext<KernelTensor> *const context,
                                      const ControlNodeParserPtr &parser) const;

  void RecordGraphInputs(const std::vector<TensorPtr> &host_tensors, const std::vector<size_t> &host_param_indexes);

  // Remove after refact.
  bool enable_prepare_case() {
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
    return !tensors_need_reprepare_[this].empty() || (has_parameter_input_ && !enable_infer_boost) || is_sub_data_;
  }

  const GraphCompilerInfo *graph_compiler_info_;
  GraphExecutionStrategy strategy_;
  GraphExecutionStrategy real_strategy_;
  HostQueueDSActorPtr host_data_source_actor_;
  HostTensorQueuePtr host_tensor_queue_;
  std::vector<std::vector<TensorPtr>> init_tensors_;

  // Record the address modified input nodes to refresh the ref node.
  std::set<AnfNode *> address_modified_input_nodes_;
  bool first_step_;
  std::vector<ShapeVector> host_tensors_;
  bool has_parameter_input_;

  // The tensor of parameter(weight) maybe update host value by Python phase and need re-prepare to sync new host value
  // to device side. 'tensors_need_reprepare_' records all tensors whose host value has updated, this HashSet will be
  // update by update value callback of tensors.
  static mindspore::HashMap<const DataPrepareActor *, mindspore::HashSet<const tensor::Tensor *>>
    tensors_need_reprepare_;
  // Record each tensor related to graph.
  static mindspore::HashMap<const tensor::Tensor *, mindspore::HashSet<const DataPrepareActor *>> tensor_with_graphs_;
  // The ref relationship of device address.
  std::map<KernelWithIndex, std::vector<KernelTensorPtr>> ref_kernel_tensors_;

  bool has_dynamic_shape_{false};

  // Global execution count for data prepare actor.
  static std::atomic<size_t> execution_count_;
  // Flatten weights needs to prepare data every step.
  bool is_sub_data_{false};
};  // namespace runtime

using DataPrepareActorPtr = std::shared_ptr<DataPrepareActor>;
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_DATA_PREPARE_ACTOR_H_
