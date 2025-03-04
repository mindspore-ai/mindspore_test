/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SUPER_KERNEL_ACTOR_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_SUPER_KERNEL_ACTOR_H_

#include <string>
#include <memory>
#include <map>
#include <utility>
#include <vector>
#include <queue>
#include <set>
#include "runtime/graph_scheduler/actor/debug_aware_actor.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/actor/kernel_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_launch_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_infer_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_resize_actor.h"
#include "runtime/hardware/device_context.h"
#include "runtime/pipeline/async_rqueue.h"
#include "ir/anf.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceAddress;
using mindspore::device::DeviceContext;

struct OutputMemoryInfo {
  size_t size;
  std::string node_full_name;
};

// Struct is used to represent the output information of node and is used to mark whether the output has been released
// when get the release position of address ptr.
struct FreeNodeInfo {
  device::DeviceContextKey context_key;
  std::string branch_name;
  bool operator<(const FreeNodeInfo &other) const {
    if (context_key.device_id_ < other.context_key.device_id_) {
      return true;
    }
    if (context_key.device_id_ > other.context_key.device_id_) {
      return false;
    }
    if (context_key.device_name_ < other.context_key.device_name_) {
      return true;
    }
    if (context_key.device_name_ > other.context_key.device_name_) {
      return false;
    }
    return branch_name < other.branch_name;
  }
};

// The Super kernel actor is used to represent the sink executing of graph which is the combination of kernels.
class SuperKernelActor : public DebugAwareActor {
 public:
  SuperKernelActor(const std::string &name, const KernelGraphPtr &graph, const std::string &graph_phase,
                   const DeviceContext *device_context, const AID &memory_manager_aid, const AID *debug_aid,
                   const AID *recorder_aid, KernelTransformType type = KernelTransformType::kSuperKernelActor);
  ~SuperKernelActor() override;

  size_t FetchInputNodePosition(const AnfNodePtr &intput_node);
  virtual void FetchInputDeviceTensor(OpContext<DeviceTensor> *const context);
  // The debug related operation interface.
  void SendDebugReq(OpContext<DeviceTensor> *const context) override;

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context) override;
  // The callback after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) override;
  // The input may come from the control actor, so need free the input memory by the dynamic ref count.
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context) override;
  bool CopyInputData(const OpContext<DeviceTensor> *context, const KernelGraphPtr &graph);

  const KernelGraphPtr &graph() const { return graph_; }

  void BuildAndLinkKernelActors();
  const std::vector<KernelActorPtr> &kernel_actors() const { return kernel_actors_; }
  const std::vector<size_t> &input_param_static_use_cnt() const { return input_params_use_cnt_; }
  const std::vector<bool> &is_input_used() const { return is_input_used_; }
  bool enable_kbk_sub_graph_execute() const { return enable_kbk_sub_graph_execute_; }

  bool enable_inline_control_flow() const { return enable_inline_control_flow_; }
  bool enable_infer_boost() const { return enable_infer_boost_; }
  const mindspore::HashMap<AnfNode *, std::vector<std::pair<size_t, size_t>>> &kernel_input_to_graph_input_indices()
    const {
    return kernel_input_to_graph_input_indices_;
  }
  const mindspore::HashMap<AnfNode *, std::vector<std::pair<size_t, std::vector<size_t>>>>
    &kernel_input_to_actor_output_indices() const {
    return kernel_input_to_actor_output_indices_;
  }
  const std::set<std::pair<size_t, ParameterInfo>> &input_params_no_user() const { return input_params_no_user_; }

  void IncreaseNewRefCounts(OpContext<DeviceTensor> *const context) override;
  // Get the release position of the device address in the graph through static analysis of the input-output
  // relationship in the graph.
  void SetFreePositionForKernelActor();
  void SetInputFreePositionForKernelActor(
    const KernelActorPtr &kernel_actor,
    const mindspore::HashMap<AnfNodePtr, device::DeviceContextKey> &kernel_to_context_key,
    const device::DeviceContextKey &graph_device_context_key,
    std::set<std::pair<KernelWithIndex, FreeNodeInfo>> *checked_nodes);
  void SetOutputFreePositionForKernelActor(
    const KernelActorPtr &kernel_actor,
    const mindspore::HashMap<AnfNodePtr, device::DeviceContextKey> &kernel_to_context_key,
    const device::DeviceContextKey &graph_device_context_key,
    std::set<std::pair<KernelWithIndex, FreeNodeInfo>> *checked_nodes);
  void GetRefCountForGraphOutput(const std::vector<AnfNodePtr> &output_data_nodes,
                                 const std::vector<DataArrowPtr> &output_data_arrows,
                                 const mindspore::HashMap<AnfNodePtr, KernelActor *> &kernel_to_actor,
                                 const std::map<uint32_t, std::vector<CNodePtr>> &inplace_groups,
                                 const std::string &actor_name);

 protected:
  void Init() override;
  void Run(OpContext<DeviceTensor> *const context) override;
  void Finalize() override;

  // The input device tensors for launch.
  std::vector<DeviceTensor *> input_device_tensors_;
  // The device tensors of graph input parameter, which used to compare the recv input data.
  std::vector<DeviceTensorPtr> node_device_tensors_;
  // The device tensors for memory alloc.
  std::vector<DeviceTensor *> memory_alloc_list_;
  // The lists of device tensors which need free by dynamic ref count, will be cleared at the end of step.
  std::queue<std::vector<DeviceTensor *>> memory_free_lists_;

 protected:
  bool CopyInputDataPersistedHandle(const DeviceContext *device_context, DeviceTensor *input_device_tensor,
                                    const DeviceTensorPtr &node_device_tensor, size_t i);

  // Generate and initialize all kernel actors by execution order of graph_ for kerkel by kernl execute a sub garph
  // mode.
  void BuildKernelActors();
  KernelActorPtr BuildInnerControlFlowActor(const CNodePtr &kernel, const DeviceContext *device_context,
                                            GraphExecutionStrategy strategy, const std::set<size_t> &ref_input_indexes,
                                            const std::set<size_t> &ref_output_indexes);

  // Parse all nodes dependence of graph_, record device tensor store key of every kernel, calculate original ref
  // count of CNode and Parameter, prepare input and heterogeneous output device address of all kernels.
  void LinkKernelActors();
  // When there is control flow in the graph, in order to control the execution of the kernel actor the relationship
  // between condition actor, as well as the relationship between condition actor and kernel actors, should be set.
  void SetRelationForControlFlow();
  void AnalyseNodesDependence(const HashMap<size_t, AnfNodePtr> &device_tensor_store_keys_map,
                              const HashMap<size_t, ParameterInfo> &parameter_indexs_map,
                              const HashMap<AnfNodePtr, std::vector<size_t>> &output_node_to_actor_output_index,
                              std::vector<std::pair<KernelActorPtr, size_t>> *param_first_used_kernel_actors);

  void LinkKernelActor(const CNodePtr &kernel, size_t input_index, const AnfNodePtr &input_node, size_t output_index);
  void LinkKernelActorByDeviceType(const CNodePtr &kernel, size_t input_index, const AnfNodePtr &input_node,
                                   size_t output_index);

  void RunGraphKernelByKernel(OpContext<DeviceTensor> *const context);
  // Need to correct current ref count or dynamic ref count by the use count of the input node(parameter) in the graph.
  // From the outside, the input device address is used only once by the super kernel actor, origin ref count only +1 in
  // compile phase.
  void CorrectRefCount(size_t input_index, DeviceTensor *device_tensor);
  void CorrectRefCountByCondition(size_t index, DeviceTensor *device_tensor,
                                  std::vector<DeviceTensor *> *memory_free_list);

  void FetchPersistentDeviceTensor();

  void UpdateMemoryTraceMangerStatus(OpContext<DeviceTensor> *const context);
  void SetTraceMemoryForKernel(const KernelActorPtr &kernel_actor, bool safe_update = false);
  // Allocate block memory for use trace memory (run by static shape) step.
  void AllocateTraceMemory(OpContext<DeviceTensor> *const context) const;
  // Free block memory for use trace memory (run by static shape) step.
  void FreeTraceMemory() const;
  void SetInputTraceMemory(const KernelActorPtr &kernel_actor) const;

  // Handle copy output for different device type kernel.
  bool CopyHeterogeneousOutput(OpContext<DeviceTensor> *const context, const KernelActorPtr &kernel_actor) const;

  void UpdateOutputAddress(const std::vector<std::pair<size_t, std::vector<size_t>>> &kernel_inputs_to_actor_outputs,
                           const KernelActorPtr &kernel_actor);

  // Launch all kernels by execution order in kernel graph: graph_.
  bool LaunchAllKernels(OpContext<DeviceTensor> *const context);

  void TrackInputMemory();

  void FetchParameterInput(const KernelActorPtr &kernel_actor, OpContext<DeviceTensor> *const context);
  void FreeInputParamWithoutUser(OpContext<DeviceTensor> *const context);
  void RecordKernelActorWeight();

  // Prepare non top cell input, such as internal parameter msg input, control flow msg input and const value.
  bool FetchMsgInputAndConstValueForKernel(KernelActor *kernel_actor, OpContext<DeviceTensor> *const context);

  void ParallelDispatchKernels(OpContext<DeviceTensor> *const context);
  // Dispatch kernel which can parallel launch.
  void DispatchParallelLaunchKernels(size_t index, OpContext<DeviceTensor> *const context);
  // Dispatch serial launch kernels: communication ops and the kernel need force resize.
  void DispatchSerialLaunchKernels(OpContext<DeviceTensor> *const context);

  void InitParallelDispatchResource();
  void PartitionParallelDispatchKernels();
  // Recreate the communication group for the communication operators, ensuring that the communication group is the
  // same for the communication operators on each concurrent thread.
  void RecreateCommunicationGroup();
  void ClearParallelDispatchResource();

  friend class GraphScheduler;
  KernelGraphPtr graph_;

  // The phase of the root graph this super actor belongs to.
  std::string graph_phase_;
  // Whether the super kernel actor is a infer 'prefill' or 'increment' graph or not.
  bool is_infer_phase_;

  // In the scheduler, check whether the parameters need to be copied after lunch. Only when the parameter has
  // the ref attribute and is directly used by the kernel in the graph, it needs to be copied.
  std::vector<bool> is_parameters_need_copy_;

  // Record the address map of ref node to copy back when running finished.
  std::map<DeviceAddress *, DeviceAddress *> ref_node_addr_map_;

  // The received input device type and format may be different from the formal parameter in the control flow scenarios,
  // so it needs to be copied from the input data to real data that graph launch needs.
  std::vector<DeviceTensorPtr> copy_input_device_tensors_;
  // Record the device address to the output node of graph.
  std::map<DeviceAddress *, OutputMemoryInfo> device_address_to_node_;

  // Record the use count of all input nodes(parameter) of graph_, use to correct current ref count in runtime.
  std::vector<size_t> input_params_use_cnt_;

  // Record the graph parameter without user.
  std::set<std::pair<size_t, ParameterInfo>> input_params_no_user_;

  std::vector<DeviceTensor *> new_memory_free_list_;

  // Record whether the input is used by kernel actor.
  std::vector<bool> is_input_used_;
  // Record every param first used kernel actor to correct the ref count.
  mindspore::HashMap<KernelActorPtr, std::vector<std::pair<size_t, size_t>>> kernel_actor_to_graph_parameters_map_;
  // Record which kernel actor should insert event when fetch parameter on non-default stream.
  mindspore::HashSet<KernelActor *> kernel_actors_insert_event_;

  // Record all parameter nodes of graph_ and their index positions in graph_'s input_nodes.
  mindspore::HashMap<AnfNode *, size_t> param_node_to_input_idx_;

  // Kernel by kernel sub graph execute mode need not send actor message.
  bool enable_kbk_sub_graph_execute_;
  bool already_fetch_persistent_device_tensor_{false};
  mindspore::HashMap<AnfNodePtr, KernelActor *> cnode_to_kernel_actor_;
  std::vector<KernelActorPtr> kernel_actors_;
  // Indices from other actor.
  mindspore::HashMap<AnfNode *, std::vector<std::pair<size_t, size_t>>> kernel_input_to_graph_input_indices_;
  mindspore::HashMap<AnfNode *, std::vector<std::pair<size_t, std::vector<size_t>>>>
    kernel_input_to_actor_output_indices_;
  SomasInfo *somas_info_;

  AID kernel_async_infer_aid_;
  AID kernel_async_resize_aid_;
  AID kernel_async_launch_aid_;

  bool enable_trace_memory_;

  // The variables for parallel dispatch kernel.
  bool enable_parallel_dispatch_{false};
  std::vector<std::vector<KernelActorPtr>> parallel_launch_kernels_;
  std::vector<KernelActorPtr> serial_launch_kernels_;
  HashMap<KernelActor *, std::vector<DeviceEventPtr>> serial_launch_kernels_to_events_;

  static size_t parallel_dispatch_num_;
  static size_t parallel_slice_num_;

  static std::vector<std::pair<size_t, void *>> streams_;
  static std::vector<DeviceEventPtr> events_;
  static std::vector<AsyncRQueuePtr> queues_;

  // Remove after input optimize simplify.
  bool first_step_for_inference_{true};
  bool enable_infer_boost_{false};

  // Whether the actor include a control flow actor.
  bool enable_inline_control_flow_{false};
};

using SuperKernelActorPtr = std::shared_ptr<SuperKernelActor>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_ACTOR_H_
