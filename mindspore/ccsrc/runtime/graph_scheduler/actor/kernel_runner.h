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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_RUNNER_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_RUNNER_H_

#include <vector>
#include <set>
#include <string>
#include <map>
#include <memory>
#include <utility>
#include "utils/hash_map.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/actor/debug_aware_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_launch_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_infer_actor.h"
#include "runtime/graph_scheduler/actor/kernel_async_resize_actor.h"
#include "runtime/hardware/device_context.h"
#include "runtime/graph_scheduler/device_tensor_store.h"
#include "common/kernel.h"
#include "ir/anf.h"
#include "ir/tensor.h"

namespace mindspore {
namespace runtime {
using mindspore::device::DeviceContext;
using mindspore::device::KernelInfo;
using mindspore::kernel::Address;
using mindspore::kernel::InputDataInfo;
using mindspore::kernel::KernelLaunchAddr;
using mindspore::kernel::KernelLaunchInfoWithStream;
using mindspore::kernel::KernelMod;
using mindspore::kernel::KernelTensor;
using mindspore::kernel::KernelTensorPtr;
using mindspore::session::SomasInfo;
using mindspore::tensor::TensorPtr;

class SuperKernelActor;

class KernelRunner {
 public:
  KernelRunner(const std::string &name, const CNodePtr &kernel, const DeviceContext *device_context,
               const AID &memory_manager_aid, const AID *debug_aid, const AID *recorder_aid,
               GraphExecutionStrategy strategy, const std::set<size_t> &modifiable_ref_input_indexes,
               const std::set<size_t> &modifiable_ref_output_indexes,
               const KernelTransformType &type = KernelTransformType::kKernelActor);

  ~KernelRunner() = default;

  // The memory related operation interface.
  void SendMemoryAllocReq(OpContext<DeviceTensor> *const context);
  void SendMemoryFreeReq(OpContext<DeviceTensor> *const context);
  // The callback after memory alloc finished.
  void OnMemoryAllocFinish(OpContext<DeviceTensor> *const context);

  const CNodePtr &kernel() const { return kernel_; }
  KernelLaunchInfoWithStream kernel_launch_info() const {
    return KernelLaunchInfoWithStream(input_kernel_tensors_, output_kernel_tensors_, workspace_kernel_tensors_,
                                      stream_);
  }
  const std::set<size_t> &modifiable_ref_input_indexes() const { return modifiable_ref_input_indexes_; }
  const std::set<size_t> &modifiable_ref_output_indexes() const { return modifiable_ref_output_indexes_; }
  bool is_dynamic_shape() const { return is_dynamic_shape_; }
  bool is_launch_skipped() const { return is_launch_skipped_; }
  bool inputs_continuous_memory() const { return inputs_continuous_memory_; }
  SomasInfo *somas_info() const { return somas_info_; }
  const std::set<size_t> &somas_graph_output_indexes() const { return somas_graph_output_indexes_; }
  size_t get_stream() const { return kernel_info_->stream_id(); }

  const mindspore::HashMap<size_t, size_t> &increase_ref_count_size() const { return increase_ref_count_size_; }
  const std::vector<bool> &is_output_kernel() const { return is_output_kernel_; }
  const std::vector<bool> &is_monad_input() const { return is_monad_input_; }
  const std::vector<size_t> &input_free_index() const { return input_free_index_; }
  const std::vector<size_t> &output_free_index() const { return output_free_index_; }
  const std::vector<bool> &depend_shape_input_list() const { return depend_shape_input_list_; }
  const std::vector<bool> &is_weight() const { return is_weight_; }

  void set_enable_async_infer(bool enable_async_infer) { enable_async_infer_ = enable_async_infer; }

  // Really do infer shape and update kernel tensor shape.
  virtual void ExecuteInferShapeTask(OpContext<DeviceTensor> *const context);
  // Really do resize kernel mod and update new size into output and workspace kernel tensors.
  virtual void ExecuteResizeKernelModTask(OpContext<DeviceTensor> *const context);
  // Really do launch kernel with memory allocate and free.
  virtual void ExecuteLaunchKernelTask(OpContext<DeviceTensor> *const context);

  void set_stream_send_actor(KernelRunner *stream_send_actor) { stream_send_actor_ = stream_send_actor; }

  void SetInputDeviceTensor(DeviceTensor *input_device_tensor, size_t input_index);

  // Set the memory address for the tensors which use the somas.
  void SetSomasMemory(OpContext<DeviceTensor> *const context) const;

  bool skip_launch_shape_related_op() const { return skip_launch_shape_related_op_; }
  void set_skip_launch_shape_related_op(bool skip_launch_shape_related_op) {
    skip_launch_shape_related_op_ = skip_launch_shape_related_op;
  }

  const std::map<size_t, std::pair<DeviceTensorPtr, std::pair<const DeviceContext *, std::vector<DeviceTensor *>>>>
    &copy_output_device_tensors() const {
    return copy_output_device_tensors_;
  }
  std::vector<DeviceTensor *> GetOutputDeviceTensors() { return output_device_tensors_; }

  // Reset state for UCE.
  void ResetState();

  const std::vector<DeviceTensor *> &workspace_device_tensors() { return workspace_device_tensors_; }
  const std::vector<DeviceTensor *> &output_device_tensors() { return output_device_tensors_; }
  const std::vector<DeviceTensor *> &input_device_tensors() { return input_device_tensors_; }

 protected:
  void Init();
  void Run(OpContext<DeviceTensor> *const context);
  void SendRecorderInfo(OpContext<DeviceTensor> *const context) const;

  // Do kernel launching in this method after 'PreLaunchKernel' and 'PostLaunchKernel'.
  virtual bool LaunchKernel(OpContext<DeviceTensor> *const context, bool is_skip_launch = false);
  // Handle the ref op, set input addr to output addr.
  virtual void UpdateRefDeviceAddress(OpContext<DeviceTensor> *const context, bool increase_ref_count);
  // Execute kernel actor multi stream produre to make sure safety of memory before kernel launch.
  void ProcessMultiStreamBeforeKernelLaunch(OpContext<DeviceTensor> *const context);
  // Execute kernel actor multi stream produre to make sure safety of memory after kernel launch.
  void ProcessMultiStreamAfterKernelLaunch(OpContext<DeviceTensor> *const context);
  // Update the output ref count of graph output kernel.
  void UpdateGraphOutputRefCount(OpContext<DeviceTensor> *const context);
  // Update the input device tensors to the memory free list.
  void UpdateMemoryFreeList(OpContext<DeviceTensor> *const context);
  // Execute infer shape, resize and launch kernel by runtime pipeline which executes by KernelAsyncInferActor,
  // KernelAsyncResizeActor and KernelAsyncLaunchActor.
  void RunWithMultiPipeline(OpContext<DeviceTensor> *const context);
  // Execute launch kernel asynchronously in KernelAsyncLaunchActor.
  void RunWithAsyncLaunchKernel(OpContext<DeviceTensor> *const context);

  // Infer shape(and type) and resize kernel mod for dynamic shape or dynamic value, and update memory size from kernel
  // mod to output/workspace device tensors.
  void InferAndUpdateDeviceTensorSize(OpContext<DeviceTensor> *const context);

  // Infer shape(and type) and resize kernel mod.
  void InferAndResize(OpContext<DeviceTensor> *const context);

  // Re-Infer shape, type and resize before kernel launch in dynamic scenarios.
  void InferShapeAndType();

  // Re-InferShape and resize before kernel launch in dynamic scenarios.
  void InferShape();

  void ResizeKernelMod();

  // Update input_device_tensors by input op data.
  void UpdateInputDeviceTensor(const OpData<DeviceTensor> *input_data, OpContext<DeviceTensor> *const context);

  // Record the output and workspace memory pointer and size to optimize memory allocate/free performance in next step.
  // Note: only use in inference case.
  void TraceDynamicMemory();

  /************ region Members of ActorBase and OpActor. ************/
  inline const AID &GetAID() const { return id; }

  AID id;

  // The op data.
  mindspore::HashMap<int, std::vector<OpData<DeviceTensor> *>> input_op_datas_;
  std::vector<DataArrowPtr> output_data_arrows_;

  // The op controls.
  mindspore::HashMap<int, std::vector<AID *>> input_op_controls_;
  std::vector<ControlArrowPtr> output_control_arrows_;
  /************ endregion ************/

  /************ region Members of AbstractActor. ************/

  // Erase input data and input controls when finish actor running.
  virtual void EraseInput(const OpContext<DeviceTensor> *context);

  // Fetch input data from the device tensor store.
  void FetchInputByTensorStore(std::vector<DeviceTensor *> *const input_device_tensors,
                               std::vector<KernelTensor *> *const input_kernel_tensors,
                               std::vector<abstract::AbstractBasePtr> *const input_kernel_tensors_for_infer,
                               std::vector<DeviceTensor *> *const memory_free_tensors,
                               OpContext<DeviceTensor> *const context) const;

  // Fetch input parameter data from the device tensor store.
  void FetchParameterByTensorStore(std::vector<DeviceTensor *> *const input_device_tensors,
                                   std::vector<KernelTensor *> *const input_kernel_tensors,
                                   std::vector<abstract::AbstractBasePtr> *const input_kernel_tensors_for_infer,
                                   std::vector<DeviceTensor *> *const memory_free_tensors,
                                   OpContext<DeviceTensor> *const context);

  // Init the member output_data_ and batch_output_data_ by output data arrows.
  void InitOutputData();
  // Update the output data before send output data.
  virtual void UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &data_arrow,
                                const AnfNodePtr &output_node, OpContext<DeviceTensor> *const context) {}
  // Send output to downstream actors to trigger running.
  virtual void SendOutput(OpContext<DeviceTensor> *const context);
  void SendOutputData(OpContext<DeviceTensor> *const context, const std::vector<AnfNodePtr> &output_data_nodes,
                      const std::vector<DataArrowPtr> &output_data_arrows,
                      const std::vector<std::pair<OpDataUniquePtr<DeviceTensor>, size_t>> &output_data_list,
                      const mindspore::HashMap<DataArrow *, size_t> &data_arrow_to_fusion_actor_indexs,
                      mindspore::HashMap<std::string, std::vector<OpData<DeviceTensor> *>> *batch_output_data);
  // Fetch the sub actor in the fusion actor by the name.
  AbstractActor *FetchSubActorInFusionActor(const std::string &sub_actor_name) const;

  virtual void IncreaseNewRefCounts(OpContext<DeviceTensor> *const context);
  virtual void IncreaseNewRefCount(const OpData<DeviceTensor> *op_data) const;

  std::vector<const DeviceContext *> device_contexts_;

  // The id of recorder actor. Send message to it for recording info.
  const AID *recorder_aid_;

  // Auto increment id for actor.
  int64_t actor_id_;

  // The output_data_nodes_ and output_data_ corresponds to the output_data_arrows_ one by one.
  std::vector<AnfNodePtr> output_data_nodes_;

  // The second of pair indicates the output data flag. See constant prefixed with kOutputDataFalg for details.
  std::vector<std::pair<OpDataUniquePtr<DeviceTensor>, size_t>> output_data_;
  // Record the fusion output index for output data arrow.
  mindspore::HashMap<DataArrow *, size_t> data_arrow_to_fusion_actor_indexs_;

  // The dependent inputs number.
  size_t input_datas_num_;
  size_t input_controls_num_;

  // The dependent messages number of actor running.
  int running_dependent_msg_num_;

  // Indicates whether the actor is in fusion actor.
  AbstractActor *parent_fusion_actor_;

  // Whether use input optimize.
  bool enable_input_optimize_;

  // The dependent device tensor stores, the dependent expression is pair<index, AnfNode>.
  // Index is the input position, AnfNode is the key of the device tensor store.
  std::vector<std::pair<size_t, AnfNodePtr>> device_tensor_store_keys_;

  // The dependent parameter stores, the dependent expression is pair<index, ParameterInfo>.
  // Index is the input position, ParameterInfo is used to fetch args and device tensor.
  std::vector<std::pair<size_t, ParameterInfo>> parameter_indexs_;

  // Used to send batch data in the message which RunBatchOpData needs, the key is the actor name of destination actor.
  mindspore::HashMap<std::string, std::vector<OpData<DeviceTensor> *>> batch_output_data_;
  mindspore::HashMap<std::string, std::vector<DataArrowPtr>> batch_output_data_arrows_;

  // When there is recursion in the graph, the actor will send data to the same stack actor multiple times. Since
  // messages are sent asynchronously between actors, there will be multiple messages that remain unprocessed in
  // the channel. In order to prevent old data from being overwritten, it is necessary to allocate a new op data,
  // and these op data will be uniformly cleared by the scheduler after the step ends.
  std::vector<OpDataUniquePtr<DeviceTensor>> to_stack_data_;
  /************ endregion ************/

  // The info of kernel.
  CNodePtr kernel_;
  bool is_dynamic_shape_;
  bool is_dynamic_value_;
  bool is_dynamic_type_;
  bool has_dynamic_;
  // Whether enable asynchronously infer shape and resize kernel mod by KernelInferActor and KernelResizeActor.
  bool enable_async_infer_;
  AID kernel_async_infer_aid_;
  AID kernel_async_resize_aid_;
  AID kernel_async_launch_aid_;
  KernelInfo *kernel_info_;
  KernelMod *kernel_mod_;

  // The device tensors for launch.
  std::vector<DeviceTensor *> input_device_tensors_;
  std::vector<DeviceTensor *> output_device_tensors_;
  std::vector<DeviceTensor *> workspace_device_tensors_;

  std::vector<DeviceTensor *> max_ref_cnt_output_list_;

  // The input kernel tensors for infer shape.
  std::vector<abstract::AbstractBasePtr> input_kernel_tensors_for_infer_;
  // The kernel tensors for resize and launch.
  std::vector<KernelTensor *> input_kernel_tensors_;
  std::vector<KernelTensor *> output_kernel_tensors_;
  std::vector<KernelTensor *> workspace_kernel_tensors_;

  // The received input device type and format may be different from the formal parameter in the control flow
  // scenarios, so it needs to be copied from the input data to real data that kernel launch needs.
  // And if the kernel has a ref input and output, the ptr should be set into the output device addresses.
  std::vector<DeviceTensorPtr> copy_input_device_tensors_;
  // This vector record the heter input device tensor which is the source of copy input device tensors, and the
  // size of this vector is same to copy input device tensors. It is used to free memory of input, when the pre
  // input is not empty, and the input should be free, the pre device tensors should be free too.
  std::vector<DeviceTensor *> pre_input_device_tensors_;
  std::map<size_t, std::pair<DeviceTensorPtr, std::pair<const DeviceContext *, std::vector<DeviceTensor *>>>>
    copy_output_device_tensors_;
  // Real data info that kernel launch needs, used to check the consistency of received input data.
  std::vector<std::shared_ptr<InputDataInfo>> real_input_data_infos_;

  // The device tensors for memory alloc and free.
  // output + workspace
  std::vector<DeviceTensor *> memory_alloc_list_;
  // input + output + workspace
  std::vector<DeviceTensor *> memory_free_list_;

  std::vector<size_t> input_free_index_;
  std::vector<size_t> output_free_index_;
  std::vector<DeviceTensor *> new_memory_free_list_;

  // depend shape input list
  std::vector<bool> depend_shape_input_list_;
  // The device tensor of external reference is not the real data of this kernel, but need add to the
  // memory_free_list_.
  std::vector<DeviceTensor *> external_reference_tensors_;

  // The information used for integration of dynamic and static memory.
  SomasInfo *somas_info_;
  // The graph output node and index use somas info.
  std::set<size_t> somas_graph_output_indexes_;
  // Task id on stream, use for events.
  std::shared_ptr<int64_t> task_id_on_stream_ = std::make_shared<int64_t>(0L);
  // Send actor ref, point to the send actor when current actor is recv actor.
  KernelRunner *stream_send_actor_{nullptr};
  // Flag for stream recv actor.
  bool is_stream_recv_actor_{false};
  // Flag for indicating if current actor is multi-thread safe, which was generate at compile time.
  bool is_multi_stream_safe_{false};
  // Record kernel actor weight input for input optimize.
  std::vector<bool> is_weight_;

 protected:
  friend class GraphScheduler;
  friend class ControlNodeScheduler;
  friend class InlineControlFlowScheduler;
  friend class SchedulerHelper;
#ifdef ENABLE_RPC_ACTOR
  friend class RpcNodeScheduler;
#endif
  friend class SuperKernelActor;

  // Init the device tensors and kernel launch info.
  void InitInputInfo();
  void InitOutputInfo();
  void InitWorkspaceInfo();
  void InitMultiStreamInfo();
  void InitIsMonadInput();

  // Fetch the device tensor for launch.
  void FetchInputDeviceTensor(OpContext<DeviceTensor> *const context);
  void FetchOutputDeviceTensor(OpContext<DeviceTensor> *const context);
  void FetchWorkspaceDeviceTensor();
  // Need copy when the data type or format between real parameters and formal parameters are inconsistent.
  void CopyInputDeviceTensor(DeviceTensor *device_tensor, size_t input_index, OpContext<DeviceTensor> *const context);
  void UpdateDeviceTensorCopyStore(DeviceTensor *const new_device_tensor, DeviceTensor *const input_device_tensor,
                                   size_t input_index);
  // The processing before kernel launch: update the info of kernel launch.
  void PreLaunchKernel(OpContext<DeviceTensor> *const context);
  // The processing after kernel launch: 1.erase input, 2.free memory, 3.send output.
  void PostLaunchKernel(OpContext<DeviceTensor> *const context);
  // Back refresh the dynamic device tensor stores that have been triggered copy.
  void RefreshDeviceTensorCopyStore(OpContext<DeviceTensor> *const context);

  void *GetSomasDevicePtr(size_t offset) const;

  // Record mem info, because async send may free device info.
  void SetMemInfoForRdr();
  void SetShapeDependInfo();
  void DispatchDebugActor(OpContext<DeviceTensor> *const context);
  bool LaunchKernelWithDebug(OpContext<DeviceTensor> *const context, const bool skip_launch);

  const AID &memory_manager_aid() const { return memory_manager_aid_; }

  // The real input number of kernel launch.
  size_t real_input_num_;

  // The execution strategy of kernel actor.
  // In pipeline mode, kernel actor executes asynchronously.
  // In step mode, kernel actor executes synchronously.
  GraphExecutionStrategy strategy_{GraphExecutionStrategy::kPipeline};

  // Record the modifiable ref indexes. Used to refresh the ref data which are modified in the running.
  std::set<size_t> modifiable_ref_input_indexes_;
  std::set<size_t> modifiable_ref_output_indexes_;

  // Whether skip the kernel launch.
  bool is_launch_skipped_;

  // Recoreded mem info.
  KernelLaunchAddr mem_info_;

  // The ignore input addresses when the kernel launch.
  std::vector<size_t> launch_ignored_inputs_;

  // Whether the inputs need continuous memory, used to check the inputs legitimacy.
  bool inputs_continuous_memory_;

  // The stream resource of the KernelActor to launch kernel.
  void *stream_{nullptr};

  bool is_multi_stream_process_skipped_{false};
  std::vector<std::pair<uint32_t, void *>> cross_stream_addresses_;

  // Flag for skipping launch shape related operator, such as RealMakeTuple.
  // RealMakeTuple --> ShapeCalc pattern: if ShapeCalc is not value depend for one input RealMakeTuple op, we can skip
  // launch this RealMakeTuple.
  bool skip_launch_shape_related_op_{false};

  // If the kernel is output of kernel graph, the ref count of output user should be add. This element record the output
  // index to the ref count size.
  mindspore::HashMap<size_t, size_t> increase_ref_count_size_;
  std::vector<bool> is_output_kernel_;
  std::vector<bool> is_monad_input_;
  bool is_mc2_kernel_{false};
  // This flag are only valid in control flow. In inline control flow, due to the existence of switch branches, some
  // kernel actors will not be executed, and the condition switch actor controls whether to execute. It points to
  // the flag of the control branch in the condition switch actor. When the switch confirms the execution of a branch,
  // it sets the flag of the branch to true to enable the actor in this branch.
  bool *is_enable_{nullptr};
  bool need_wait_pipeline_{false};

  // The id of debug actor. Send message to it for debug.
  const AID *debug_aid_;
  const AID *profiler_aid_;

  // The id of memory manager actor. Send message to it for alloc and free memory.
  const AID memory_manager_aid_;
};

using KernelRunnerPtr = std::shared_ptr<KernelRunner>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_ACTOR_KERNEL_RUNNER_H_
