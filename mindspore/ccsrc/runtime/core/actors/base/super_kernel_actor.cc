/**
 * Copyright 2021-2025 Huawei Technologies Co., Ltd
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

#include "runtime/core/actors/base/super_kernel_actor.h"
#include <set>
#include <algorithm>
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "runtime/core/graph_scheduler/base/scheduler_helper.h"
#include "runtime/core/actors/base/output_actor.h"
#include "runtime/core/actors/base/memory_manager_actor.h"
#include "runtime/core/actors/base/debug_actor.h"
#include "runtime/core/actors/control_flow/condition_switch_runner.h"
#include "runtime/core/actors/control_flow/condition_gather_runner.h"
#include "include/runtime/pipeline/task/batch_launch_kernel_task.h"
#include "runtime/core/graph_executor/pipeline/runtime_pipeline.h"
#include "include/runtime/utils/runtime_conf/runtime_conf.h"
#include "runtime/core/graph_scheduler/base/parameter_store.h"
#include "runtime/core/graph_executor/kernel_capture/graph_capture_manager.h"
#include "async/async.h"
#include "utils/llm_manager.h"
#include "utils/log_adapter.h"
#include "utils/ms_exception.h"
#include "utils/trace_base.h"
#include "op_def/framework_ops.h"
#include "pybind_api/gil_scoped_long_running.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "include/cluster/topology/collective_manager.h"
#include "include/backend/debug/execute_order_tracker/execute_order_tracker.h"
#include "tools/error_handler/error_config.h"

namespace mindspore {
namespace runtime {
size_t SuperKernelActor::parallel_dispatch_num_ = 2;
size_t SuperKernelActor::parallel_slice_num_ = 4;

std::vector<std::pair<size_t, void *>> SuperKernelActor::streams_;
std::vector<DeviceEventPtr> SuperKernelActor::events_;
std::vector<DeviceEventPtr> SuperKernelActor::events_to_default_stream_;
std::vector<AsyncRQueuePtr> SuperKernelActor::queues_;

static SpinLock spin_lock;
static std::mutex mtx;
bool SuperKernelActor::already_allocate_trace_memory_ = false;

namespace {
inline void UpdateShape(const AnfNodePtr &input_node, const KernelTensorPtr &node_device_kernel_tensor,
                        const KernelTensorPtr &input_kernel_tensor, const KernelTransformType &type) {
  MS_EXCEPTION_IF_NULL(input_node);
  MS_EXCEPTION_IF_NULL(node_device_kernel_tensor);
  MS_EXCEPTION_IF_NULL(input_kernel_tensor);
  if (type != KernelTransformType::kSuperKernelActor || input_node->cast<ParameterPtr>()->has_dynamic_shape()) {
    // For dynamic shape in sub graph sink and any type parameter, the input size should be updated.
    node_device_kernel_tensor->device_address()->SetSize(input_kernel_tensor->device_address()->GetSize());
    // Update Shape.
    node_device_kernel_tensor->SetShape(input_kernel_tensor->GetShape()->Clone());
  }
}

inline bool InputDataNoNeedCopy(const AnfNodePtr &input_node, const KernelTensorPtr &input_kernel_tensor,
                                const KernelTensorPtr &node_kernel_tensor, const KernelTransformType &type) {
  if (input_kernel_tensor == nullptr) {
    return true;
  }
  auto node_device_tensor = node_kernel_tensor->device_address().get();
  auto input_device_tensor = input_kernel_tensor->device_address().get();
  MS_EXCEPTION_IF_NULL(node_device_tensor);
  if (input_device_tensor == nullptr) {
    return true;
  }

  if (input_device_tensor == node_device_tensor) {
    (void)input_kernel_tensor->TouchSyncHandler();
    return true;
  }

  UpdateShape(input_node, node_kernel_tensor, input_kernel_tensor, type);

  if (TEST_FLAG(node_kernel_tensor->flag(), device::kDeviceAddressFlagNotUsed) ||
      input_device_tensor->GetPtr() == node_device_tensor->GetPtr()) {
    return true;
  }

  return false;
}

bool IsOnlyDependShape(const CNodePtr &kernel, size_t input_index) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  static const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  if (enable_infer_boost) {
    return false;
  }

  const auto &only_depend_shape_attr = common::AnfAlgo::GetCNodePrimitiveAttr(kernel, kAttrOnlyDependShape);
  if (only_depend_shape_attr != nullptr) {
    const auto &only_depend_shape = GetValue<std::vector<bool>>(only_depend_shape_attr);
    if (input_index < only_depend_shape.size() && only_depend_shape[input_index]) {
      return true;
    }
  }
  return false;
}

void SetParamFirstUsedKernelActors(
  size_t graph_input_index, size_t actor_input_index, KernelRunnerPtr *kernel_actor,
  std::vector<std::pair<KernelRunnerPtr, size_t>> *param_first_used_kernel_actors,
  mindspore::HashMap<size_t, mindspore::HashMap<size_t, KernelRunnerPtr>> *param_first_used_actors_on_stream) {
  if (!EnableInputOptimize()) {
    return;
  }
  if (graph_input_index >= (*param_first_used_kernel_actors).size()) {
    MS_LOG(EXCEPTION) << "Index " << graph_input_index << " is out of range size "
                      << (*param_first_used_kernel_actors).size();
  }
  // Record non default stream first used graph parameters.
  if (kernel_actor != nullptr && (*kernel_actor) != nullptr && (*kernel_actor)->get_stream() != kDefaultStreamIndex) {
    const auto &iter = (*param_first_used_actors_on_stream).find(graph_input_index);
    if (iter == (*param_first_used_actors_on_stream).end()) {
      (*param_first_used_actors_on_stream)[graph_input_index].emplace((*kernel_actor)->get_stream(), (*kernel_actor));
    } else {
      const auto &actor_sets_with_stream = iter->second;
      const auto &actor_set_iter = actor_sets_with_stream.find((*kernel_actor)->get_stream());
      if (actor_set_iter == actor_sets_with_stream.end()) {
        (*param_first_used_actors_on_stream)[graph_input_index].emplace((*kernel_actor)->get_stream(), (*kernel_actor));
      }
    }
  }

  if ((*param_first_used_kernel_actors)[graph_input_index].first == nullptr) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    (*param_first_used_kernel_actors)[graph_input_index].first = *kernel_actor;
    (*param_first_used_kernel_actors)[graph_input_index].second = actor_input_index;
  }
}

void CollectStreamFirstUsedParamKernelActors(
  mindspore::HashMap<size_t, mindspore::HashMap<size_t, KernelRunnerPtr>> *param_first_used_actors_on_stream) {
  if (!EnableInputOptimize()) {
    return;
  }
  for (const auto &iter : *param_first_used_actors_on_stream) {
    const auto &stream_with_kernel_actors = iter.second;
    for (const auto &stream_with_actor_iter : stream_with_kernel_actors) {
      MS_EXCEPTION_IF_NULL(stream_with_actor_iter.second);
      stream_with_actor_iter.second->set_insert_input_event(true);
    }
  }
}

void RecordInputParamsWithoutUser(const KernelGraphPtr &graph,
                                  const HashMap<size_t, ParameterInfo> &parameter_indexs_map,
                                  const std::vector<size_t> &input_params_use_cnt,
                                  std::set<std::pair<size_t, ParameterInfo>> *input_params_no_user) {
  if (!EnableInputOptimize()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(input_params_no_user);
  const auto &input_nodes = graph->input_nodes();
  size_t input_num = input_nodes.size();
  for (size_t i = 0; i < input_num; ++i) {
    if (input_nodes[i]->isa<Parameter>() &&
        (common::AnfAlgo::IsParameterWeight(input_nodes[i]->cast<ParameterPtr>()))) {
      continue;
    }
    if (input_params_use_cnt.at(i) == 0) {
      const auto &parameter_index_iter = parameter_indexs_map.find(i);
      if (parameter_index_iter != parameter_indexs_map.end()) {
        input_params_no_user->emplace(i, parameter_index_iter->second);
      }
    }
  }
}

void CalculateParameterUsedTimes(const std::map<std::pair<size_t, size_t>, size_t> &parameter_used_times) {
  auto enable_graph_capture = GraphCaptureManager::GetInstance().GetEnableGraphCapture();
  if (!EnableInputOptimize() || (!EnableParallelDispatchKernel() && !enable_graph_capture)) {
    return;
  }
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  for (const auto &used_times_iter : parameter_used_times) {
    auto outer_index = used_times_iter.first.first;
    auto inner_index = used_times_iter.first.second;
    auto times = used_times_iter.second;
    // After enabling the capture graph feature, the value here is 2, which can force the locked process to be used
    // during fetch parameter, ensuring no conflicts occur in concurrent calls.
    if (enable_graph_capture) {
      times = kIndex2;
    }
    // If the parameter only used in this graph, but used by multiple actors when parallel dispatch.
    // Correct the parameter use times.
    // If not parallel dispatch and only used in this graph, there is no concurrently used.
    if (!graph_parameter_store->IsConcurrentlyUse(outer_index, inner_index)) {
      graph_parameter_store->SetParameterUsedTimes(outer_index, inner_index, times);
    }
  }
}
}  // namespace

SuperKernelActor::~SuperKernelActor() { ClearParallelDispatchResource(); }

void SuperKernelActor::Finalize() { ClearParallelDispatchResource(); }

void SuperKernelActor::ClearParallelDispatchResource() {
  if (enable_parallel_dispatch_) {
    std::unique_lock<std::mutex> lock(mtx);
    if (!queues_.empty()) {
      for (auto &q : queues_) {
        q->WorkerJoin();
      }
      queues_.clear();
    }
    if (!events_.empty()) {
      events_.clear();
    }
    if (!events_to_default_stream_.empty()) {
      events_to_default_stream_.clear();
    }
    if (!serial_launch_kernels_to_events_.empty()) {
      serial_launch_kernels_to_events_.clear();
    }
    if (!parallel_launch_kernels_.empty()) {
      parallel_launch_kernels_.clear();
    }
    if (!serial_launch_kernels_.empty()) {
      serial_launch_kernels_.clear();
    }
  }
}

void SuperKernelActor::Init() {
  MS_EXCEPTION_IF_NULL(graph_);
  // Check device contexts number.
  if (device_contexts_.size() != runtime::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  // Set the number of actor running dependent messages.
  running_dependent_msg_num_ = SizeToInt(input_datas_num_ + input_controls_num_);

  if (enable_parallel_dispatch_) {
    InitParallelDispatchResource();
  }

  // Init the output data.
  InitOutputData();
  if (output_data_arrows_.size() != output_data_nodes_.size()) {
    MS_LOG(EXCEPTION) << "The size of output data arrows is not equal to the output data nodes.";
  }
  if (output_data_arrows_.size() != output_data_.size()) {
    MS_LOG(EXCEPTION) << "The size of output data arrows is not equal to the output data.";
  }
  for (size_t i = 0; i < output_data_arrows_.size(); ++i) {
    auto &data_arrow = output_data_arrows_[i];
    auto &output_node = output_data_nodes_[i];
    auto data = output_data_[i].first.get();
    MS_EXCEPTION_IF_NULL(data_arrow);
    MS_EXCEPTION_IF_NULL(output_node);
    MS_EXCEPTION_IF_NULL(data);
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, IntToSize(data_arrow->from_output_index_), false);
    data->data_ = kernel_tensor;
  }

  if (enable_kbk_sub_graph_execute_) {
    // 1. Don't cache DeviceAddress of Parameter node into node_device_tensors_ on PyNative mode.
    // 2. Ignore the operator of SuperKernelActor for O2(GE) mode.
    return;
  }

  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(graph_->output());
  for (const auto &origin_output_with_index : output_with_indexs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(origin_output_with_index);
    const auto &output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    if (output_node->isa<CNode>() && (!HasAbstractMonad(output_node))) {
      auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, output_with_index.second, false);
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto device_address = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_address);
      if (kernel_tensor->is_ptr_persisted() || graph_->is_dynamic_shape()) {
        MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
          << "Actor:" << GetAID() << " skip alloc memory for device address:" << device_address
          << " is persist:" << kernel_tensor->is_ptr_persisted() << " is dynamic shape:" << graph_->is_dynamic_shape()
          << " output node:" << output_node->DebugString();
        continue;
      }
      // Free the ptr in device address of output node.
      if (device_address->GetPtr() != nullptr) {
        MS_LOG(INFO) << "Output node:" << output_node->DebugString() << " has a default ptr, maybe a mem leak.";
        device_address->set_ptr(nullptr);
      }
      if (IsSkippedLaunch()) {
        device_address_to_node_[device_address.get()] = {device_address->GetSize(), output_node->fullname_with_scope()};
      }
      memory_alloc_list_.emplace_back(kernel_tensor);
    }
  }

  // Check whether the parameter needs to be copied out.
  node_kernel_tensors_.resize(graph_->input_nodes().size());
  is_parameters_need_copy_.resize(graph_->input_nodes().size());
  copy_input_kernel_tensors_.resize(graph_->input_nodes().size());
  for (size_t i = 0; i < graph_->input_nodes().size(); ++i) {
    const auto &input_node = graph_->input_nodes()[i];
    MS_EXCEPTION_IF_NULL(input_node);
    node_kernel_tensors_[i] = AnfAlgo::GetOutputKernelTensor(input_node, 0, false);
    if (!common::AnfAlgo::HasAbstractRef(input_node)) {
      is_parameters_need_copy_[i] = false;
      continue;
    }
    // If the parameter has ref attribute and is directly used by the kernel in the graph, it needs to be copied.
    is_parameters_need_copy_[i] = true;
  }
}

void SuperKernelActor::InitParallelDispatchResource() {
  MS_EXCEPTION_IF_CHECK_FAIL(streams_.empty(),
                             "streams_ is not empty, parallel dispatch resource is already initialized.");
  streams_.resize(parallel_dispatch_num_);
  for (size_t i = 0; i < parallel_dispatch_num_; i++) {
    if (!device_contexts_[0]->device_res_manager_->CreateStream(&(streams_[i].first))) {
      MS_LOG(EXCEPTION) << "Create stream failed.";
    }
    streams_[i].second = device_contexts_[0]->device_res_manager_->GetStream(streams_[i].first);
    MS_EXCEPTION_IF_NULL(streams_[i].second);
  }

  MS_EXCEPTION_IF_CHECK_FAIL(events_.empty(),
                             "events_ is not empty, parallel dispatch resource is already initialized.");
  // New one more for sync between default stream and last launch stream;
  for (size_t i = 0; i < parallel_dispatch_num_ * parallel_slice_num_ + 1; i++) {
    auto event = device_contexts_[0]->device_res_manager_->CreateEventWithFlag(false, false, false);
    MS_EXCEPTION_IF_NULL(event);
    events_.push_back(event);
  }

  MS_EXCEPTION_IF_CHECK_FAIL(
    events_to_default_stream_.empty(),
    "events_to_default_stream_ is not empty, parallel dispatch resource is already initialized.");
  for (size_t i = 0; i < streams_.size(); i++) {
    auto event = device_contexts_[0]->device_res_manager_->CreateEventWithFlag(false, false, true);
    MS_EXCEPTION_IF_NULL(event);
    events_to_default_stream_.push_back(event);
  }

  MS_EXCEPTION_IF_CHECK_FAIL(queues_.empty(),
                             "queues_ is not empty, parallel dispatch resource is already initialized.");
  for (size_t i = 0; i < parallel_dispatch_num_; i++) {
    auto queue = std::make_unique<AsyncRQueue>(std::string("batch_launch_") + std::to_string(i),
                                               runtime::kThreadWaitLevel::kLevelDevice);
    MS_EXCEPTION_IF_NULL(queue);
    queue->SetSpin(false);
    queues_.push_back(std::move(queue));
  }

  const size_t kEventNum = 2;
  for (auto &kernel_actor : serial_launch_kernels_) {
    serial_launch_kernels_to_events_[kernel_actor.get()] = std::vector<DeviceEventPtr>(kEventNum, nullptr);
  }

  for (auto &item : serial_launch_kernels_to_events_) {
    auto &event_array = item.second;
    for (size_t i = 0; i < event_array.size(); i++) {
      auto event = device_contexts_[0]->device_res_manager_->CreateEventWithFlag(false, false, false);
      MS_EXCEPTION_IF_NULL(event);
      event_array[i] = event;
    }
  }
}

size_t SuperKernelActor::FetchInputNodePosition(const AnfNodePtr &intput_node) {
  MS_EXCEPTION_IF_NULL(intput_node);
  MS_EXCEPTION_IF_NULL(graph_);

  auto &input_nodes = graph_->input_nodes();
  const auto &iter = find(input_nodes.begin(), input_nodes.end(), intput_node);
  if (iter == input_nodes.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, intput_node) << "Invalid input node:" << intput_node->fullname_with_scope();
  }
  return iter - input_nodes.begin();
}

void SuperKernelActor::FetchInputDeviceTensor(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                  "Invalid device context for super kernel actor:" + GetAID().Name());
  }
  std::vector<KernelTensorPtr> memory_free_list;
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      MS_EXCEPTION_IF_NULL(input_data->data_);
      size_t index = IntToSize(input_data->index_);
      if (index >= input_kernel_tensors_.size()) {
        std::string error_info = "Invalid input index:" + std::to_string(index) +
                                 " total:" + std::to_string(input_kernel_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      input_kernel_tensors_[index] = input_data->data_;
      MS_EXCEPTION_IF_NULL(input_kernel_tensors_[index]);

      if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
        auto output_address = input_kernel_tensors_[index]->device_address().get();
        MS_EXCEPTION_IF_NULL(output_address);
        MS_LOG(WARNING) << "Need Profile Memory, Memory use, actor name: " << GetAID().Name()
                        << ", kernel graph: " << graph_->ToString() << ", device address class ptr: " << output_address
                        << ", device address size: " << input_kernel_tensors_[index]->device_address()->GetSize()
                        << ", device address addr: " << input_kernel_tensors_[index]->device_address()->GetPtr()
                        << ", index: " << index;
      }

      if (!enable_kbk_sub_graph_execute_ || ActorDispatcher::enable_use_trace_memory()) {
        if (input_data->data_->new_ref_count() != SIZE_MAX) {
          (void)memory_free_list.emplace_back(input_data->data_);
        }

        continue;
      }
    }
    if (!enable_kbk_sub_graph_execute_ || ActorDispatcher::enable_use_trace_memory()) {
      memory_free_lists_.push(memory_free_list);
      return;
    }
  }
}

void SuperKernelActor::Run(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "Super Kernel actor:" << GetAID() << " start run.";
  if (enable_kbk_sub_graph_execute_) {
    try {
      return RunGraphKernelByKernel(context);
    } catch (const std::exception &e) {
      if (context->error_info_.empty()) {
        MsException::Instance().SetException();
        std::string error_info =
          "Run graph[" + std::to_string(graph_->graph_id()) + "] by kernek by kernel failed, exception: " + e.what();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
    }
    return;
  }
  if (NeedRunMemTracker()) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), "SuperKernelActor", graph_->ToString(),
                                                   true);
  }
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Invalid device context for super kernel actor:" + GetAID().Name());
  }
  MS_LOG(INFO) << "Super kernel actor(" << GetAID().Name()
               << ") launches graph: " << std::to_string(graph_->graph_id());
  if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
    MS_LOG(WARNING) << "Need Profile Memory, launch actor name: " << GetAID().Name()
                    << ", kernel graph: " << graph_->ToString();
  }
  if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
    MS_LOG(INFO) << "Run failed and early stop.";
    return;
  }
  FetchInputDeviceTensor(context);
  if (!already_fetch_persistent_device_tensor_) {
    FetchPersistentDeviceTensor();
    already_fetch_persistent_device_tensor_ = is_infer_phase_;
  }

  TrackInputMemory();

  if (memory_alloc_list_.size() > 0) {
    for (auto &kernel_tensor : memory_alloc_list_) {
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto device_tensor = kernel_tensor->device_address().get();
      MS_EXCEPTION_IF_NULL(device_tensor);
      if (kernel_tensor->IsNotNeedAlloc()) {
        continue;
      }
      if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
        auto &info = device_address_to_node_[device_tensor];
        auto output_address = reinterpret_cast<std::uintptr_t>(device_tensor);
        MS_LOG(WARNING) << "Need Profile Memory, Memory need allocated, actor name: " << GetAID().Name()
                        << ", kernel graph: " << graph_->ToString() << ", node: " << info.node_full_name
                        << ", device address class ptr: " << output_address << ", device address size: " << info.size;
      }
      if (NeedRunMemTracker()) {
        device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, GetAID().Name(),
                                                       memory::mem_pool::MemType::kGraphOutput,
                                                       device_tensor->GetSize(), device_tensor);
      }
    }
    SendMemoryAllocReq(context);
  } else {
    OnMemoryAllocFinish(context);
  }
  if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
    MS_LOG(WARNING) << "Need Profile Memory, end launch, actor name: " << GetAID().Name()
                    << ", kernel graph: " << graph_->ToString();
  }
  MS_VLOG(VL_RUNTIME_FRAMEWORK_ACTOR) << "Super Kernel actor:" << GetAID() << " end run.";
}

void SuperKernelActor::FetchPersistentDeviceTensor() {
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto input_kernel_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                      device_contexts_[0]->GetDeviceType());
    // Ge backend maybe nullptr.
    if (input_kernel_tensor == nullptr) {
      MS_LOG(DEBUG) << "Failed get device tensor for node:" << device_tensor_store_key.second->DebugString()
                    << " index:" << device_tensor_store_key.first;
      continue;
    }

    size_t index = device_tensor_store_key.first;
    input_kernel_tensors_[index] = input_kernel_tensor;
  }
}

void SuperKernelActor::UpdateMemoryTraceMangerStatus(OpContext<KernelTensor> *const context) {
  MemoryTraceManager::GetInstance().PickMemoryTrackInfoForGraph(graph_->graph_id());
  if (!ActorDispatcher::enable_static_shape()) {
    ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kMemoryAlloc, GetAID().Name());

    const std::shared_ptr<mindspore::HashMap<CNodePtr, std::vector<KernelMemoryTraceBlockPtr>>> &all_kernel_block_info =
      MemoryTraceManager::GetInstance().GetAllKernelBlocksnfo();
    MS_EXCEPTION_IF_NULL(all_kernel_block_info);

    if (!all_kernel_block_info->empty()) {
      size_t kernel_num = kernel_actors_.size();
      for (size_t i = 0; i < kernel_num; i++) {
        const auto &kernel_actor = kernel_actors_[i];
        if (kernel_actor == nullptr) {
          continue;
        }

        const auto &kernel = kernel_actor->kernel_;
        MS_EXCEPTION_IF_NULL(kernel);

        const auto &iter = all_kernel_block_info->find(kernel);
        if (iter == all_kernel_block_info->end()) {
          MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL)
            << "Not found kernel block info for kernel: " << kernel->fullname_with_scope()
            << ", is output kernel: " << kernel_actor->is_output_kernel_;
        } else {
          const auto &kernel_mem_block = iter->second;
          for (auto &block : kernel_mem_block) {
            MS_EXCEPTION_IF_NULL(block);
            if (block->mem_type_ == kOutputMem) {
              auto &kernel_tensor = kernel_actor->output_kernel_tensors_.at(block->index_);
              kernel_tensor->set_device_ptr(nullptr);
            } else {
              kernel_actor->workspace_kernel_tensors_.at(block->index_)->set_device_ptr(nullptr);
            }
          }
        }
      }
    }

    // First step for dynamic shape, need to record memory trace.
    MemoryTraceManager::GetInstance().ClearExpiredCache();
    static const size_t memory_block_size = 3000;
    MemoryTraceManager::GetInstance().ReserveKernelMemoryBlocks(memory_block_size, device_contexts_[0]);
  } else {
    // Not first step for dynamic shape, use record trace memory.
    if (!already_allocate_trace_memory_) {
      AllocateTraceMemory(context);
    }
  }
}

void SuperKernelActor::SetTraceMemoryForKernel(const KernelRunnerPtr &kernel_actor, bool safe_update) {
  const auto &kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(kernel);

  // Allocate trace memory for static memory step.
  const std::shared_ptr<mindspore::HashMap<CNodePtr, std::vector<KernelMemoryTraceBlockPtr>>> &all_kernel_block_info =
    MemoryTraceManager::GetInstance().GetAllKernelBlocksnfo();
  MS_EXCEPTION_IF_NULL(all_kernel_block_info);
  const auto &iter = all_kernel_block_info->find(kernel);
  if (iter == all_kernel_block_info->end()) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Not found kernel block info for kernel: " << kernel->fullname_with_scope()
                                         << ", is output kernel: " << kernel_actor->is_output_kernel_;
  } else {
    const auto &kernel_mem_block = iter->second;
    const auto &merge_blocks_with_device_context = MemoryTraceManager::GetInstance().GetMergeBlocks();
    MS_EXCEPTION_IF_NULL(merge_blocks_with_device_context);
    const auto &merge_blocks = merge_blocks_with_device_context->at(kernel_actor->real_output_device_context_);
    for (auto &block : kernel_mem_block) {
      MS_EXCEPTION_IF_NULL(block);
      void *ptr = merge_blocks.at(block->in_memory_trace_block_index_)->start_ + block->offset_in_memory_trace_block_;
      MS_EXCEPTION_IF_NULL(ptr);
      if (block->mem_type_ == kOutputMem) {
        if (!safe_update) {
          auto &kernel_tensor = kernel_actor->output_kernel_tensors_.at(block->index_);
          kernel_tensor->set_device_ptr(ptr);
        } else {
          auto &kernel_tensor = kernel_actor->output_kernel_tensors_.at(block->index_);
          std::lock_guard<SpinLock> lock(block->lock_);
          if (kernel_tensor->device_ptr() != ptr) {
            kernel_tensor->set_device_ptr(ptr);
          }
        }
      } else {
        kernel_actor->workspace_kernel_tensors_.at(block->index_)->set_device_ptr(ptr);
      }
    }
  }
}

void SuperKernelActor::SetInputTraceMemory(const KernelRunnerPtr &kernel_actor) const {
  const auto &merge_blocks_with_device_context = MemoryTraceManager::GetInstance().GetMergeBlocks();
  MS_EXCEPTION_IF_NULL(merge_blocks_with_device_context);

  const auto &kernel_tensor_to_kernel_mem_blocks = MemoryTraceManager::GetInstance().GetKernelTensorToMemBlocksInfo();
  MS_EXCEPTION_IF_NULL(kernel_tensor_to_kernel_mem_blocks);

  for (auto &input_kernel_tensor : kernel_actor->input_kernel_tensors_) {
    const auto &iter = kernel_tensor_to_kernel_mem_blocks->find(input_kernel_tensor.get());
    if (iter == kernel_tensor_to_kernel_mem_blocks->end()) {
      continue;
    }
    auto &kernel_mem_block = iter->second;
    const auto &merge_blocks = merge_blocks_with_device_context->at(kernel_mem_block->device_context_);
    void *ptr = merge_blocks.at(kernel_mem_block->in_memory_trace_block_index_)->start_ +
                kernel_mem_block->offset_in_memory_trace_block_;

    std::lock_guard<SpinLock> lock(kernel_mem_block->lock_);
    if (input_kernel_tensor->device_ptr() != ptr) {
      input_kernel_tensor->set_device_ptr(ptr);
    }
  }
}

void SuperKernelActor::AllocateTraceMemory(OpContext<KernelTensor> *const context) const {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kMemoryAlloc, GetAID().Name());
  const auto &merge_blocks_with_device_context = MemoryTraceManager::GetInstance().GetMergeBlocks();
  MS_EXCEPTION_IF_NULL(merge_blocks_with_device_context);
  for (auto &item : *merge_blocks_with_device_context) {
    const auto &device_context = item.first;
    MS_EXCEPTION_IF_NULL(device_context);
    const auto &merge_blocks = item.second;
    for (auto &block : merge_blocks) {
      MS_EXCEPTION_IF_NULL(block);
      static const size_t kMemoryAlignSize = 1024;
      void *block_addr = device_context->device_res_manager_->AllocateMemory(block->size_ + kMemoryAlignSize);
      if (block_addr == nullptr) {
        SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *context, *device_context,
                                                    GetAID().Name(), block->size_);
      }
      block->start_ = reinterpret_cast<uint8_t *>(block_addr);
    }
  }
  already_allocate_trace_memory_ = true;
}

void SuperKernelActor::FreeTraceMemory() const {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kMemoryFree, GetAID().Name());
  const auto &merge_blocks_with_device_context = MemoryTraceManager::GetInstance().GetMergeBlocks();
  MS_EXCEPTION_IF_NULL(merge_blocks_with_device_context);
  for (auto &item : *merge_blocks_with_device_context) {
    const auto &device_context = item.first;
    MS_EXCEPTION_IF_NULL(device_context);
    const auto &merge_blocks = item.second;
    for (auto &block : merge_blocks) {
      MS_EXCEPTION_IF_NULL(block);
      device_context->device_res_manager_->FreeMemory(block->start_);
    }
  }
  already_allocate_trace_memory_ = false;
}

bool SuperKernelActor::CopyHeterogeneousOutput(OpContext<KernelTensor> *const context,
                                               const KernelRunnerPtr &kernel_actor) const {
  if (!WaitRuntimePipelineFinish(context, kernel_actor->kernel_mod_->kernel_name())) {
    return false;
  }

  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kCopyData,
                            kernel_actor->kernel_mod_->kernel_name());
  for (const auto &output_index_to_copy_kt : kernel_actor->copy_output_kernel_tensors_) {
    const auto &output_index = output_index_to_copy_kt.first;
    const auto &dest_kernel_tensor = output_index_to_copy_kt.second.first;
    const auto &dest_device_address = dest_kernel_tensor->device_address();
    const auto &dest_device_context = output_index_to_copy_kt.second.second.first;
    const auto &src_kernel_tensor = kernel_actor->output_kernel_tensors_.at(output_index);
    MS_EXCEPTION_IF_NULL(src_kernel_tensor);
    const auto &src_device_address = src_kernel_tensor->device_address();
    const auto &ref_output_kernel_tensors = output_index_to_copy_kt.second.second.second;

    if (src_device_address != nullptr && !IsContiguousStorage(src_device_address->GetTensorStorageInfo())) {
      MS_LOG(ERROR) << "Not support non-contiguous heter output:" << src_kernel_tensor->ToString()
                    << " for actor:" << GetAID();
      return false;
    }
    if (kernel_actor->is_dynamic_shape_) {
      // For dynamic shape case.
      dest_kernel_tensor->SetType(src_kernel_tensor->GetType()->Clone());
      dest_kernel_tensor->SetShape(src_kernel_tensor->GetShape()->Clone());
      dest_kernel_tensor->set_size(src_kernel_tensor->size());
    }

    // Allocate memory.
    if (dest_kernel_tensor->device_ptr() != nullptr) {
      if (ref_output_kernel_tensors.empty()) {
        MS_LOG_WITH_NODE(EXCEPTION, kernel_actor->kernel_)
          << "Memory leak detected in copy output device address for kernel: "
          << kernel_actor->kernel_->fullname_with_scope() << " device address:" << dest_kernel_tensor->ToString();
      }
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Free heter output address:" << dest_kernel_tensor->ToString() << " for actor:" << kernel_actor->GetAID();
      dest_device_context->device_res_manager_->FreeMemory(dest_device_address.get());
    }
    std::vector<KernelTensorPtr> mem_alloc_list = {dest_kernel_tensor};
    MemoryManagerActor::GetInstance()->AllocateMemory(&mem_alloc_list, dest_device_context, context,
                                                      kernel_actor->GetAID());
    if (IsRunningFailed(context)) {
      // Maybe allocate memory failed, early stop to run graph.
      return false;
    }
    if (!SyncAllStreamForDeviceAddress(dest_device_address, src_device_address) ||
        !SyncCopy(dest_kernel_tensor.get(), src_kernel_tensor.get(), kDefaultStreamIndex)) {
      MS_LOG(ERROR) << "Copy for heterogeneous output failed, kernel actor: " << kernel_actor->GetAID().Name()
                    << ", output index: " << output_index << ", dest device address: " << dest_device_address
                    << ", src device address: " << src_device_address;
      return false;
    }
    if (!ref_output_kernel_tensors.empty()) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Add kernel tensor copy store for :" << src_kernel_tensor->ToString() << " and "
        << dest_kernel_tensor->ToString() << " for actor:" << GetAID();
      KernelTensorCopyStore::GetInstance().Insert(src_kernel_tensor.get(), dest_kernel_tensor.get());
    }
  }
  if (kernel_actor->new_memory_free_list_.size() > 0) {
    MS_LOG(DEBUG) << "Free device ptr after heter copy for actor:" << kernel_actor->GetAID();
    kernel_actor->SendMemoryFreeReq(context);
  }
  return true;
}

void SuperKernelActor::UpdateOutputAddress(
  const std::vector<std::pair<size_t, std::vector<size_t>>> &kernel_inputs_to_actor_outputs,
  const KernelRunnerPtr &kernel_actor) {
  for (const auto &pair : kernel_inputs_to_actor_outputs) {
    size_t kernel_input_index = pair.first;
    KernelTensorPtr real_input = kernel_actor->input_kernel_tensors_[kernel_input_index];
    MS_EXCEPTION_IF_NULL(real_input);
    const std::vector<size_t> &actor_output_indices = pair.second;
    real_input->IncreaseNewRefCount(GetAID().Name(), actor_output_indices.size());
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Increase ref count to:" << real_input->new_ref_count() << " increase size:" << actor_output_indices.size() - 1
      << " for kernel tensor:" << real_input->ToString() << " in actor:" << GetAID();
    for (auto actor_output_index : actor_output_indices) {
      auto data = output_data_[actor_output_index].first.get();
      MS_EXCEPTION_IF_NULL(data);
      data->data_ = real_input;
    }
  }
}

void SuperKernelActor::FetchParameterInput(const KernelRunnerPtr &kernel_actor, OpContext<KernelTensor> *const context,
                                           size_t stream_id) {
  if (!enable_input_optimize_) {
    return;
  }

  for (const auto &parameter_index : kernel_actor->parameter_indexs()) {
    size_t kernel_input_index = parameter_index.first;
    bool is_first_user = kernel_actor->is_first_used_params_[kernel_input_index];
    auto kernel_tensor = FetchParameter(parameter_index.second, kernel_actor->GetAID(), is_first_user, stream_id,
                                        enable_parallel_dispatch_);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Actor: " << kernel_actor->GetAID().Name() << ", input index: " << kernel_input_index
      << ", kernel tensor info: " << kernel_tensor->ToString()
      << " super kernel actor context:" << device_contexts_[0]->device_context_key().ToString()
      << " kernel actor context:" << kernel_actor->device_contexts()[0]->device_context_key().ToString();

    kernel_actor->SetInputDeviceTensor(kernel_tensor, kernel_input_index);
    if (is_first_user) {
      HandleFirstUserInputMemoryFree(kernel_actor, kernel_input_index);
    }
    kernel_actor->CopyInputDeviceTensor(kernel_actor->input_kernel_tensors_[kernel_input_index], kernel_input_index,
                                        context, in_increment_);
  }
}

void SuperKernelActor::HandleFirstUserInputMemoryFree(const KernelRunnerPtr &kernel_actor, size_t kernel_input_index) {
  if (ActorDispatcher::enable_use_trace_memory()) {
    if (kernel_actor->input_kernel_tensors_[kernel_input_index]->new_ref_count() != SIZE_MAX) {
      std::lock_guard<SpinLock> locker(spin_lock);
      memory_free_lists_.back().emplace_back(kernel_actor->input_kernel_tensors_[kernel_input_index]);
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Add memory free list for trace kernel tensor:"
        << kernel_actor->input_kernel_tensors_[kernel_input_index]->ToString() << " in actor:" << GetAID();
    }
  }
}

void SuperKernelActor::FreeInputParamWithoutUser(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, "FreeInputParamWithoutUser");
  if (enable_input_optimize_) {
    for (const auto &iter : input_params_no_user_) {
      auto kernel_tensor = FetchParameter(iter.second, GetAID());
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      if (kernel_tensor->new_ref_count() != SIZE_MAX) {
        // No user for this input in graph.
        MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
          << "Free ref count for no used parameter:" << iter.second.first.first->DebugString()
          << " inner index:" << iter.second.first.second << " out index:" << iter.second.second
          << " kernel tensor:" << kernel_tensor->ToString()
          << " device context:" << device_contexts_[0]->device_context_key().ToString() << " for actor:" << GetAID();
        MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(kernel_tensor.get(), device_contexts_[0],
                                                                GetAID().Name());
      }
    }
  }
}

bool SuperKernelActor::FetchMsgInputAndConstValueForKernel(KernelRunner *kernel_actor,
                                                           OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  const auto &kernel = kernel_actor->kernel();

  // 1 Prepare received input from other actors.
  const auto &iter = kernel_input_to_graph_input_indices_.find(kernel.get());
  if (iter != kernel_input_to_graph_input_indices_.end()) {
    std::vector<std::pair<size_t, size_t>> &input_to_graph_input_indices = iter->second;
    for (const auto &item : input_to_graph_input_indices) {
      MS_LOG(DEBUG) << "kernel:" << iter->first->fullname_with_scope() << " graph input index:" << item.second
                    << " kernel input index:" << item.first << " for actor:" << GetAID()
                    << " graph:" << graph_->ToString();
      kernel_actor->SetInputDeviceTensor(input_kernel_tensors_[item.second], item.first);
      kernel_actor->memory_free_list_[item.first] = input_kernel_tensors_[item.second];
      kernel_actor->CopyInputDeviceTensor(input_kernel_tensors_[item.second], item.first, context, in_increment_);
    }
  }
  // 2. Prepare const value.
  if (!kernel_actor->device_tensor_store_keys_.empty()) {
    // Collect the inputs from device tensor store.
    kernel_actor->FetchInputByTensorStore(&kernel_actor->input_launch_tensors_, &kernel_actor->input_kernel_tensors_,
                                          &kernel_actor->input_kernel_tensors_for_infer_,
                                          &kernel_actor->memory_free_list_, context);
    if (IsRunningFailed(context)) {
      return false;
    }
  }
  return true;
}

void SuperKernelActor::AsyncLaunchKernelByCondition(OpContext<KernelTensor> *const context, KernelRunner *kernel_actor,
                                                    bool hp_mode) {
  // Check high performance condition in SuperKernelActor.
  if (hp_mode) {
    if (EnableRuntimeNewPipeline()) {
      auto launch_task = [context, kernel_actor]() {
        KernelAsyncLaunchActor::GetInstance()->LaunchKernelV2HP(context, kernel_actor);
      };
      RuntimePipeline::GetInstance().launch_queue()->Push(std::move(launch_task));
    } else {
      Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernelV2HP, context, kernel_actor);
    }
  } else {
    if (EnableRuntimeNewPipeline()) {
      auto launch_task = [context, kernel_actor]() {
        KernelAsyncLaunchActor::GetInstance()->LaunchKernelV2(context, kernel_actor);
      };
      RuntimePipeline::GetInstance().launch_queue()->Push(std::move(launch_task));
    } else {
      Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernelV2, context, kernel_actor);
    }
  }
}

void SuperKernelActor::SyncDispatchKernel(OpContext<KernelTensor> *const context, KernelRunner *kernel_actor,
                                          bool hp_mode) {
  if (!ActorDispatcher::enable_static_shape()) {
    kernel_actor->device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
    // Infer shape and resize for dynamic shape or dynamice value case when disable runtime multi pipeline.
    kernel_actor->InferAndUpdateDeviceTensorSize(context);
  } else if (LLMManager::GetInstance().need_force_resize(kernel_actor->kernel_mod_->kernel_name()) ||
             kernel_actor->is_dynamic_value_) {
    kernel_actor->ResizeKernelMod();
    kernel_actor->FetchOutputDeviceTensor(context);
    kernel_actor->FetchWorkspaceDeviceTensor();
  }

  // Check high performance condition in SuperKernelActor.
  if (hp_mode) {
    KernelAsyncLaunchActor::GetInstance()->LaunchKernelV2HP(context, kernel_actor);
  } else {
    KernelAsyncLaunchActor::GetInstance()->LaunchKernelV2(context, kernel_actor);
  }
}

bool SuperKernelActor::LaunchKernelForCaptureGraph(OpContext<KernelTensor> *const context,
                                                   const KernelRunnerPtr &kernel_actor, size_t index, bool is_graph) {
  kernel_actor->UpdateRefDeviceAddress(context, true);
  kernel_actor->UpdateGraphOutputRefCount(context);
  kernel_actor->UpdateMemoryFreeList(context);
  auto kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_mod = kernel_actor->kernel_mod();
  MS_EXCEPTION_IF_NULL(kernel_mod);
  // Update output device address for Parameter as graph output case.
  const auto &input_to_output_iter = kernel_input_to_actor_output_indices_.find(kernel.get());
  if (input_to_output_iter != kernel_input_to_actor_output_indices_.end()) {
    const auto &kernel_inputs_to_actor_outputs = input_to_output_iter->second;
    UpdateOutputAddress(kernel_inputs_to_actor_outputs, kernel_actor);
  }
  if (LLMManager::GetInstance().need_force_resize(kernel_mod->kernel_name())) {
    kernel_actor->ResizeKernelMod();
    kernel_actor->FetchOutputDeviceTensor(context);
    kernel_actor->FetchWorkspaceDeviceTensor();
  }

  if (!kernel_actor->memory_alloc_list_.empty()) {
    kernel_actor->SendMemoryAllocReqHP(context, GraphCaptureManager::GetInstance().GetStreamId());
  }

  if (!kernel_actor->LaunchKernelHP(context, IsSkippedLaunch(kernel, nullptr))) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel) << "#umsg#Kernel error:#umsg#Launch kernel failed: " +
                                             kernel->fullname_with_scope()
                                        << trace::DumpSourceLines(kernel);
  }

  if (kernel_actor->is_dynamic_shape() && kernel_mod->IsNeedUpdateOutputShapeAndSize()) {
    kernel_mod->UpdateOutputShapeAndSize(kernel_actor->input_launch_tensors_, kernel_actor->output_launch_tensors_);
  }
  // In capture mode, If launch single op, record the related infos, include the device_ptr, shape and size.
  if (!is_graph) {
    GraphCaptureManager::GetInstance().RecodeInfoForSingleOp(kernel_actor, index);
  }

  if (kernel_actor->new_memory_free_list_.size() > 0 && kernel_actor->copy_output_kernel_tensors_.empty()) {
    kernel_actor->SendMemoryFreeReq(context);
  }
  return true;
}

bool SuperKernelActor::LaunchKernelForReplayGraph(OpContext<KernelTensor> *const context,
                                                  const KernelRunnerPtr &kernel_actor, size_t index) {
  auto kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(kernel);
  auto kernel_mod = kernel_actor->kernel_mod();
  MS_EXCEPTION_IF_NULL(kernel_mod);

  // In replay mode, before launch single op, recover all the infos.
  GraphCaptureManager::GetInstance().RecoverInfoForSingleOp(kernel_actor, index);

  kernel_actor->ResizeKernelMod();
  kernel_actor->FetchOutputDeviceTensor(context);
  kernel_actor->FetchWorkspaceDeviceTensor();

  if (!kernel_actor->LaunchKernelHP(context, IsSkippedLaunch(kernel, nullptr))) {
    MS_LOG_WITH_NODE(EXCEPTION, kernel) << "#umsg#Kernel error:#umsg#Launch kernel failed: " +
                                             kernel->fullname_with_scope()
                                        << trace::DumpSourceLines(kernel);
  }

  if (kernel_actor->is_dynamic_shape() && kernel_mod->IsNeedUpdateOutputShapeAndSize()) {
    MS_LOG(EXCEPTION) << "Replay single op meet shape changed!";
  }
  return true;
}

bool SuperKernelActor::LaunchKernel(OpContext<KernelTensor> *const context, const KernelRunnerPtr &kernel_actor,
                                    bool hp_mode, bool sync_run) {
  if (!ActorDispatcher::enable_use_trace_memory()) {
    kernel_actor->UpdateRefDeviceAddress(context, true);
    kernel_actor->UpdateGraphOutputRefCount(context);
    kernel_actor->UpdateMemoryFreeList(context);
  }

  // Update output device address for Parameter as graph output case.
  const auto &input_to_output_iter = kernel_input_to_actor_output_indices_.find(kernel_actor->kernel().get());
  if (input_to_output_iter != kernel_input_to_actor_output_indices_.end()) {
    const auto &kernel_inputs_to_actor_outputs = input_to_output_iter->second;
    UpdateOutputAddress(kernel_inputs_to_actor_outputs, kernel_actor);
  }

  // 1. Allocate somas memory or cached memory for this kernel.
  kernel_actor->SetSomasMemory(context);
  if (ActorDispatcher::enable_use_trace_memory()) {
    SetTraceMemoryForKernel(kernel_actor);
  }

  // 2. Async/Sync Run Infer or Launch
  if (ActorDispatcher::enable_runtime_multi_pipeline() && !ActorDispatcher::enable_static_shape() && !sync_run) {
    // If the kernel need user data and is dynamic, maybe need input kernel's output user data to infer shape, this
    // value depend case can not handle in KernelTensor auto sync phase currently.
    if (kernel_actor->kernel_mod_->need_user_data() && kernel_actor->has_dynamic_) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL)
        << "Begin wait runtime pipeline for kernel: " << kernel_actor->kernel_->fullname_with_scope();
      if (!WaitRuntimePipelineFinish(context, kernel_actor->kernel_mod_->kernel_name())) {
        return false;
      }
      MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL)
        << "End wait runtime pipeline for kernel: " << kernel_actor->kernel_->fullname_with_scope();
    }

    // Push run task to pipeline.
    // Note: dynamic value or static shape also need push task into infer actor to make sure correct kernel
    // execution order.
    if (EnableRuntimeNewPipeline()) {
      auto infer_task = [context, kernel_actor_ptr = kernel_actor.get(), high_perf_mode = hp_mode]() {
        KernelAsyncInferActor::GetInstance()->InferShapeV2(context, kernel_actor_ptr, high_perf_mode);
      };
      RuntimePipeline::GetInstance().infer_queue()->Push(std::move(infer_task));
    } else {
      Async(kernel_async_infer_aid_, &KernelAsyncInferActor::InferShapeV2, context, kernel_actor.get(), hp_mode);
    }

    // The computed depend kernel should wait output shape update after kernel launch.
    if (kernel_actor->kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL)
        << "Begin wait runtime pipeline for kernel: " << kernel_actor->kernel_->fullname_with_scope();
      if (!WaitRuntimePipelineFinish(context, kernel_actor->kernel_mod_->kernel_name())) {
        return false;
      }
      MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL)
        << "End wait runtime pipeline for kernel: " << kernel_actor->kernel_->fullname_with_scope();
    }
  } else if (ActorDispatcher::enable_async_launch_kernel() && !sync_run) {
    auto &llm_manager = LLMManager::GetInstance();
    // Execution Rules:
    // 1. The force resize operator rely on the resize of runtime framework.
    // 2. Dynamic shapes in non-dynamic-to-static require infer shape and resize.
    // 3. Dynamic values ​​in dynamic-to-static scenarios require resize and not need to infer shape.
    if (llm_manager.need_force_resize(kernel_actor->kernel_mod_->kernel_name())) {
      kernel_actor->ResizeKernelMod();
      kernel_actor->FetchOutputDeviceTensor(context);
      kernel_actor->FetchWorkspaceDeviceTensor();
    } else if (!ActorDispatcher::enable_static_shape()) {
      kernel_actor->device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
      // Infer shape and resize for dynamic shape or dynamice value case when disable runtime multi pipeline.
      kernel_actor->InferAndUpdateDeviceTensorSize(context);
    } else if (kernel_actor->is_dynamic_value_) {
      kernel_actor->ResizeKernelMod();
      kernel_actor->FetchOutputDeviceTensor(context);
      kernel_actor->FetchWorkspaceDeviceTensor();
    }

    AsyncLaunchKernelByCondition(context, kernel_actor.get(), hp_mode);

    // The computed depend kernel should wait output shape update after kernel launch.
    if (kernel_actor->kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL)
        << "Begin wait runtime pipeline for kernel: " << kernel_actor->kernel_->fullname_with_scope();
      if (!WaitRuntimePipelineFinish(context, kernel_actor->kernel_mod_->kernel_name())) {
        MS_LOG(INFO) << "Run failed and early stop for kernel: " << kernel_actor->kernel_->fullname_with_scope();
        return false;
      }
      MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL)
        << "End wait runtime pipeline for kernel: " << kernel_actor->kernel_->fullname_with_scope();
    }
  } else {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Sync launch kernel actor:" << kernel_actor->GetAID()
                                         << " in actor:" << GetAID();
    SyncDispatchKernel(context, kernel_actor.get(), hp_mode);
  }

  return true;
}

bool SuperKernelActor::LaunchAllKernels(OpContext<KernelTensor> *const context, bool hp_mode) {
  size_t kernel_num = kernel_actors_.size();
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel_actor = kernel_actors_[i];
    if (kernel_actor == nullptr) {
      continue;
    }
    if (enable_inline_control_flow_ && !*(kernel_actor->is_enable_)) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Skip launch kernel for actor:" << kernel_actor->GetAID();
      continue;
    }

    if (NeedRunMemTracker()) {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, kernel_actor->GetAID().Name(),
                                                     kernel_actor->kernel()->fullname_with_scope(),
                                                     kernel_actor->kernel()->func_graph()->ToString(), false);
    }
    // 1. Prepare input data for kernel
    // 1.1. Prepare top cell parameter input.
    FetchParameterInput(kernel_actor, context);
    // 1.2. Prepare non top cell input, such as internal parameter msg input, control flow msg input and const value.
    if (!FetchMsgInputAndConstValueForKernel(kernel_actor.get(), context)) {
      return false;
    }

    // 2. Async/Sync Run Infer or Launch
    if (!LaunchKernel(context, kernel_actor, hp_mode)) {
      return false;
    }

    if (enable_inline_control_flow_ && kernel_actor->need_wait_pipeline_) {
      if (!WaitRuntimePipelineFinish(context, kernel_actor->kernel_mod_->kernel_name())) {
        MS_LOG(INFO) << "Run failed and early stop.";
        return false;
      }
      MS_LOG(DEBUG) << "Condition switch actor:" << kernel_actor->GetAID() << " wait succeed.";
    }

    // 3. Copy for heterogeneous output device address if need.
    if (kernel_actor->copy_output_kernel_tensors_.empty()) {
      continue;
    }
    if (!CopyHeterogeneousOutput(context, kernel_actor)) {
      MS_INTERNAL_EXCEPTION(RuntimeError)
        << "Copy for heterogeneous output failed, kernel actor: " << kernel_actor->GetAID().Name();
    }
  }

  return true;
}

void SuperKernelActor::DispatchParallelLaunchKernels(size_t index, OpContext<KernelTensor> *const context) {
  if (index >= parallel_dispatch_num_) {
    MS_LOG(EXCEPTION) << "Invalid index: " << index << ", expected less than: " << parallel_dispatch_num_;
  }
  device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
  size_t real_stream_id = streams_[index].first;
  void *real_stream = streams_[index].second;

  for (size_t inner_index = 0; inner_index < parallel_slice_num_; inner_index++) {
    events_[index + inner_index * parallel_dispatch_num_]->WaitEventWithoutReset(real_stream_id);

    const auto &kernel_actors = parallel_launch_kernels_[index + inner_index * parallel_dispatch_num_];
    for (auto &kernel_actor : kernel_actors) {
      if (!kernel_actor) {
        continue;
      }

      auto commu_iter = serial_launch_kernels_to_events_.find(kernel_actor.get());
      if (commu_iter != serial_launch_kernels_to_events_.end()) {
        ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, "RecordWaitEvent");
        const auto &event_array = commu_iter->second;
        auto &record_event = event_array[0];
        auto &wait_event = event_array[1];
        record_event->RecordEvent(real_stream_id);
        wait_event->WaitEventWithoutReset(real_stream_id);
        continue;
      }

      const auto &kernel = kernel_actor->kernel_;
      FetchParameterInput(kernel_actor, context, real_stream_id);
      if (!FetchMsgInputAndConstValueForKernel(kernel_actor.get(), context)) {
        MS_LOG(EXCEPTION) << "Failed to fetch input and const value for kernel: " << kernel->fullname_with_scope();
      }
      SetTraceMemoryForKernel(kernel_actor, true);
      SetInputTraceMemory(kernel_actor);
      if (!kernel_actor->max_ref_cnt_output_list_.empty()) {
        // Allocate dynamic memory for graph output.
        MemoryManagerActor::GetInstance()->AllocateMemory(&(kernel_actor->max_ref_cnt_output_list_),
                                                          kernel_actor->device_contexts_[0], context,
                                                          kernel_actor->GetAID());
      }

      if (!kernel_actor->is_launch_skipped_) {
        MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Begin launch kernel: " << kernel_actor->kernel_->fullname_with_scope();
        uint64_t start_time = 0;
        PROFILER_START(start_time);
        auto ret = kernel_actor->kernel_mod_->Launch(kernel_actor->input_launch_tensors_,
                                                     kernel_actor->workspace_launch_tensors_,
                                                     kernel_actor->output_launch_tensors_, real_stream);
        PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunch,
                     kernel_actor->GetAID().Name(), false, kernel_actor->input_launch_tensors_);
        if (!ret) {
          MS_LOG(EXCEPTION) << "Launch kernel failed, kernel name: " << kernel_actor->kernel_->fullname_with_scope();
        }
        MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "End launch kernel: " << kernel_actor->kernel_->fullname_with_scope();
      }
    }

    events_[index + inner_index * parallel_dispatch_num_ + 1]->RecordEvent(real_stream_id);
  }
}

void SuperKernelActor::DispatchSerialLaunchKernels(OpContext<KernelTensor> *const context) {
  auto *default_stream = device_contexts_[0]->device_res_manager_->GetStream(0);
  for (auto &kernel_actor : serial_launch_kernels_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    auto comm_iter = serial_launch_kernels_to_events_.find(kernel_actor.get());
    if (comm_iter == serial_launch_kernels_to_events_.end()) {
      MS_LOG(EXCEPTION) << "Not find kernel actor : " << kernel_actor->kernel()->fullname_with_scope();
    }

    const auto &kernel = kernel_actor->kernel_;
    FetchParameterInput(kernel_actor, context, 0);
    if (!FetchMsgInputAndConstValueForKernel(kernel_actor.get(), context)) {
      MS_LOG(EXCEPTION) << "Failed to fetch input and const value for kernel: " << kernel->fullname_with_scope();
    }
    SetTraceMemoryForKernel(kernel_actor, true);
    SetInputTraceMemory(kernel_actor);
    if (!kernel_actor->max_ref_cnt_output_list_.empty()) {
      // Allocate dynamic memory for graph output.
      MemoryManagerActor::GetInstance()->AllocateMemory(
        &(kernel_actor->max_ref_cnt_output_list_), kernel_actor->device_contexts_[0], context, kernel_actor->GetAID());
    }

    const auto &event_array = comm_iter->second;
    auto &wait_event = event_array[0];
    wait_event->WaitEventWithoutReset(0);

    auto &llm_manager = LLMManager::GetInstance();
    bool need_force_resize =
      llm_manager.need_force_resize(kernel_actor->kernel_mod_->kernel_name()) || kernel_actor->is_dynamic_value_;
    if (need_force_resize) {
      kernel_actor->ResizeKernelMod();
      kernel_actor->FetchOutputDeviceTensor(nullptr);
      kernel_actor->FetchWorkspaceDeviceTensor();
    }

    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Begin serial launch kernel: "
                                         << kernel_actor->kernel_->fullname_with_scope();
    uint64_t start_time = 0;
    PROFILER_START(start_time);
    auto ret =
      kernel_actor->kernel_mod_->Launch(kernel_actor->input_launch_tensors_, kernel_actor->workspace_launch_tensors_,
                                        kernel_actor->output_launch_tensors_, default_stream);
    PROFILER_END(start_time, runtime::ProfilerModule::kKernel, runtime::ProfilerEvent::kKernelLaunch,
                 kernel_actor->GetAID().Name(), false, kernel_actor->input_launch_tensors_);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Launch kernel failed, kernel name: " << kernel_actor->kernel_->fullname_with_scope();
    }
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "End serial launch kernel: "
                                         << kernel_actor->kernel_->fullname_with_scope();

    auto &record_event = event_array[1];
    record_event->RecordEvent(0);
  }
}

void SuperKernelActor::ParallelDispatchKernels(OpContext<KernelTensor> *const context) {
  MS_LOG(INFO) << "Begin parallel dispatch kernels for graph: " << graph_->ToString();
  if (EnableRuntimeNewPipeline()) {
    // Not need launch queue work in spin in parallel launch step, which doesn't push launch task into launch queue.
    RuntimePipeline::GetInstance().launch_queue()->Pause();
    ActorDispatcher::set_enable_async_launch_kernel(false);
  }
  // Record a event to default stream to notify parallel launch kernels execute on other stream.
  events_.front()->RecordEvent(0);

  // Dispatch kernel which can parallel launch.
  for (size_t i = 0; i < parallel_dispatch_num_; i++) {
    const auto &queue = queues_[i];
    queue->Push(
      std::make_shared<BatchLaunchKernelTask>([this, i, context]() { DispatchParallelLaunchKernels(i, context); }));
  }

  // Dispatch serial launch kernels: communication ops and the kernel need force resize.
  DispatchSerialLaunchKernels(context);

  for (auto &q : queues_) {
    GilReleaseWithCheck release_gil;
    q->Wait();
  }

  // The default stream need wait all parallel launch kernel execute finish.
  events_.back()->WaitEventWithoutReset(0);
  // Reset all event for reuse.
  {
    ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPostLaunch, "ResetAllEvent");
    for (auto &e : events_) {
      e->ResetEvent();
    }
    for (auto &item : serial_launch_kernels_to_events_) {
      for (auto &e : item.second) {
        e->ResetEvent();
      }
    }
  }
  // It must be ensured that the reset at the tail is executed within the step, or an event create via
  // aclrtCreateEventExWithFlag is inserted between the parallel stream and the default stream.
  size_t parallel_stream_size = streams_.size();
  for (size_t i = 0; i < parallel_stream_size; ++i) {
    auto &e = events_to_default_stream_[i];
    e->RecordEvent(streams_[i].first);
    e->WaitEventWithoutReset(0);
  }
  MS_LOG(INFO) << "End parallel dispatch kernels for graph: " << graph_->ToString();
}

void SuperKernelActor::RunGraphKernelByKernel(OpContext<KernelTensor> *const context) {
  // Mode check for dynamic shape, async launch and runtime multi pipeline.
  if (!ActorDispatcher::enable_async_launch_kernel()) {
    std::string error_info =
      "Runtime pipeline optimization is disabled, failed to execute graph kernel by kernel mode.";
    MS_LOG(ERROR) << "Run graph failed, graph id: " << std::to_string(graph_->graph_id()) << ". " << error_info;
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
  if (graph_->is_dynamic_shape() && !ActorDispatcher::enable_runtime_multi_pipeline()) {
    MS_LOG(DEBUG)
      << "Run dynamic shape graph: " << graph_->ToString()
      << ", but does not enable runtime multi pipeline, maybe the thread number of actor is not greater than 3.";
  }
  if (!graph_->is_dynamic_shape()) {
    ActorDispatcher::set_enable_static_shape(false);
  }
  if (memory_free_lists_.empty()) {
    memory_free_lists_.push({});
  }

  // 1. Fetch input data for this kernel graph and correct current ref count for input device address.
  FetchInputDeviceTensor(context);
  FreeInputParamWithoutUser(context);

  // 2. Allocate somas memory for graph
  if ((somas_info_ != nullptr) && (somas_info_->whole_block_size_ != 0)) {
    MemoryManagerActor::GetInstance()->AllocateSomasMemory(somas_info_, device_contexts_[0], context, GetAID());
    if (IsRunningFailed(context)) {
      // Maybe allocate memory failed, early stop to run graph.
      MS_LOG(INFO) << "Run failed and early stop to run graph: " << graph_->ToString();
      return;
    }
  }

  MS_LOG(DEBUG) << "Run graph: " << graph_->ToString()
                << ", enable capture graph global: " << GraphCaptureManager::GetInstance().GetEnableGraphCapture();

  if (enable_trace_memory_ && graph_->is_dynamic_shape() && (graph_phase_.find("increment") != std::string::npos)) {
    MS_LOG(DEBUG) << "Enable trace memory for increment inference graph: " << graph_->graph_id()
                  << ", phase: " << graph_phase_;
    UpdateMemoryTraceMangerStatus(context);
    if (IsRunningFailed(context)) {
      // Maybe allocate memory failed, early stop to run graph.
      MS_LOG(INFO) << "Run failed and early stop to run graph: " << graph_->ToString();
      return;
    }
  }

  device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
  // 3. Launch all kernels by execution order.
  if (ActorDispatcher::enable_parallel_dispatch_kernel_for_cur_actor_set()) {
    ParallelDispatchKernels(context);
  } else {
    if (!GraphCaptureManager::GetInstance().IsInit() && GraphCaptureManager::GetInstance().GetEnableGraphCapture() &&
        (graph_phase_.find("increment") != std::string::npos)) {
      MS_INTERNAL_EXCEPTION(RuntimeError)
        << "It seems that you did not set up the ACL graph before the first step. You need to "
           "adjust the timing of enabling the ACL graph at the first step.";
    }
    bool need_capture_graph = enable_capture_graph_ && !GraphCaptureManager::GetInstance().HasCapturedGraph() &&
                              ActorDispatcher::enable_static_shape() &&
                              !GraphCaptureManager::GetInstance().IsExceedMaxCaptureCount();
    bool need_replay_graph = enable_capture_graph_ && GraphCaptureManager::GetInstance().HasCapturedGraph();
    GraphCaptureManager::GetInstance().SetInReplay(need_replay_graph);

    if (!need_capture_graph && !need_replay_graph) {
      if (!LaunchAllKernels(context, is_high_perf_mode_ && IsHighPerfModeAtExec())) {
        MS_INTERNAL_EXCEPTION(RuntimeError)
          << "Launch kernels by execution order failed for graph: " << graph_->ToString();
      }
    } else if (need_capture_graph) {
      GraphCaptureManager::GetInstance().Initialize(device_contexts_[0]);
      GraphCaptureManager::GetInstance().FetchAllInputsBeforeCaptureGraph(context, kernel_actors_, &memory_free_lists_);
      if (!GraphCaptureManager::GetInstance().LaunchAllKernelsWithCapture(
            context, kernel_actors_, this, is_high_perf_mode_ && IsHighPerfModeAtExec())) {
        MS_INTERNAL_EXCEPTION(RuntimeError)
          << "Launch kernels with capture graph failed for graph: " << graph_->ToString();
      }
    } else if (need_replay_graph) {
      ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kGraphLaunch, GetAID().Name(), false);
      auto result = std::async(std::launch::async, [this]() {
        return GraphCaptureManager::GetInstance().CheckParameterNotChange(static_cast<size_t>(0));
      });
      GraphCaptureManager::GetInstance().UpdateFixAddressBeforeReplayGraph(0, &memory_free_lists_);
      if (!GraphCaptureManager::GetInstance().LaunchAllKernelsWithReplayGraph(
            context, kernel_actors_, this, is_high_perf_mode_ && IsHighPerfModeAtExec())) {
        MS_INTERNAL_EXCEPTION(RuntimeError) << "Replay graph failed for graph: " << graph_->ToString();
      }

      if (!result.get()) {
        MS_LOG(EXCEPTION) << "check parameter error when replay capture graph, it's seems like the parameter device "
                             "address has changed.";
      }
      GraphCaptureManager::GetInstance().ResetInfoForSingleOp(kernel_actors_);
    }
  }

  if (((somas_info_ != nullptr) && (somas_info_->whole_block_size_ != 0)) ||
      ActorDispatcher::enable_trace_dynamic_memory() || ActorDispatcher::enable_use_trace_memory()) {
    WaitRuntimePipelineFinish(context, GetAID().Name());
  }

  // 4. Free somas or cached memory for graph.
  if ((somas_info_ != nullptr) && (somas_info_->whole_block_size_ != 0)) {
    MemoryManagerActor::GetInstance()->FreeSomasMemory(somas_info_, device_contexts_[0], context, GetAID());
  }

  if (ActorDispatcher::enable_trace_dynamic_memory()) {
    // Record and analyse the memory trace of this step, use to optimize the memory manage performance.
    ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kMemoryFree, GetAID().Name());
    MemoryTraceManager::GetInstance().MergeBlocks();
  }

  if (ActorDispatcher::enable_use_trace_memory()) {
    // Free block memory for use trace memory (run by static shape) step.
    if (!GraphCaptureManager::GetInstance().GetEnableGraphCapture()) {
      FreeTraceMemory();
    }
  }

  // Free input data for use trace memory (run by static shape) step.
  PostRun(context);
}

void SuperKernelActor::SendMemoryAllocReq(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                  "Invalid device context for super kernel actor:" + GetAID().Name());
  }
  sort(memory_alloc_list_.begin(), memory_alloc_list_.end(), [](const KernelTensorPtr a, const KernelTensorPtr b) {
    MS_EXCEPTION_IF_NULL(a);
    MS_EXCEPTION_IF_NULL(b);
    MS_EXCEPTION_IF_NULL(a->device_address());
    MS_EXCEPTION_IF_NULL(b->device_address());
    return a->device_address()->GetSize() > b->device_address()->GetSize();
  });
  if (ActorDispatcher::is_memory_allocation_sync()) {
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &memory_alloc_list_,
                              device_contexts_[0], context, GetAID());
    OnMemoryAllocFinish(context);
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &memory_alloc_list_,
                          device_contexts_[0], context, GetAID());
  }
}

void SuperKernelActor::OnMemoryAllocFinish(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);
  if (IsRunningFailed(context)) {
    MS_LOG(INFO) << "Running failed in actor:" << GetAID().Name();
    return;
  }
  {
    ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
    if (!CopyInputData(context, graph_)) {
      std::string error_info = "Copy the input data failed, graph id: " + std::to_string(graph_->graph_id());
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }

  try {
    if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                    "Invalid device context for super kernel actor:" + GetAID().Name());
    }
    MS_LOG(EXCEPTION) << "Launch graph error.";
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info = "Launch graph exception, graph id: " + std::to_string(graph_->graph_id());
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  {
    ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPostLaunch, GetAID().Name());
    for (auto item : ref_node_addr_map_) {
      MS_EXCEPTION_IF_NULL(item.first);
      MS_EXCEPTION_IF_NULL(item.second);
      MS_LOG(INFO) << "The input ref node copy back from address: " << item.first->device_ptr()
                   << " to address: " << item.second->device_ptr() << ".";
      if (!SyncCopy(item.second.get(), item.first.get(), kDefaultStreamIndex)) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Copy data failed.");
      }
    }
    ref_node_addr_map_.clear();
  }

  // Debug actor is blocked, must wait debug actor callback message to process continue.
  if (debug_aid_ != nullptr) {
    SendDebugReq(context);
    return;
  }
  PostRun(context);
}

void SuperKernelActor::SendDebugReq(OpContext<KernelTensor> *const context) {
  running_dependent_msg_num_ = 1;
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                  "Invalid device context for super kernel actor:" + GetAID().Name());
  }
  OnDebugFinish(context);
}

bool SuperKernelActor::CopyInputDataPersistedHandle(const DeviceContext *device_context,
                                                    const KernelTensorPtr &input_kernel_tensor,
                                                    const KernelTensorPtr &node_kernel_tensor, size_t i) {
  auto &input_device_tensor = input_kernel_tensor->device_address();
  auto &node_device_tensor = node_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(node_device_tensor);
  if ((input_device_tensor->GetDeviceType() == node_device_tensor->GetDeviceType()) &&
      AnfAlgo::IsEquivalentFormat(input_kernel_tensor->format(), node_kernel_tensor->format())) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Not need copy for device tensor:" << node_device_tensor << " ptr:" << node_device_tensor->GetPtr()
      << " index:" << i << " for actor:" << GetAID();
    // Set the ptr from input_device_tensor and set mem pool false to avoid memory double management for
    // supporting zero copy.
    if (type_ != KernelTransformType::kSuperKernelActor) {
      node_device_tensor->set_ptr(input_device_tensor->GetMutablePtr());
    } else {
      node_device_tensor->set_ptr(input_kernel_tensor->GetValidPtr(input_device_tensor->stream_id()));
    }
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Actor:" << GetAID() << "set need sync flag from:" << input_device_tensor << " to:" << node_device_tensor
      << " sync user data handler:" << node_kernel_tensor->need_sync_user_data();
    node_device_tensor->set_from_mem_pool(false);
    // continue
    return true;
  }
  if (device_context->GetDeviceType() != node_device_tensor->GetDeviceType()) {
    device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {node_device_tensor->GetDeviceType(), node_device_tensor->device_id()});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  }

  if (copy_input_kernel_tensors_[i] == nullptr) {
    MS_EXCEPTION_IF_NULL(node_kernel_tensor);
    const auto new_kernel_tensor = SchedulerHelper::CloneKernelTensorWithDeviceInfo(node_kernel_tensor, device_context);
    MS_EXCEPTION_IF_NULL(new_kernel_tensor);
    new_kernel_tensor->set_device_ptr(nullptr);

    copy_input_kernel_tensors_[i] = new_kernel_tensor;
    MS_LOG(DEBUG) << "Create new kernel tensor:" << copy_input_kernel_tensors_[i] << " index:" << i
                  << " for actor:" << GetAID();
  }
  auto copy_kernel_tensor = copy_input_kernel_tensors_[i];
  MS_EXCEPTION_IF_NULL(copy_kernel_tensor);
  auto &copy_device_tensor = copy_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(copy_device_tensor);
  copy_kernel_tensor->set_user_data(node_kernel_tensor->user_data());
  copy_kernel_tensor->set_need_sync_user_data(node_kernel_tensor->need_sync_user_data());
  if (copy_device_tensor->GetPtr() == nullptr) {
    if (!device_context->device_res_manager_->AllocateMemory(copy_device_tensor.get())) {
      MS_LOG(ERROR) << "Device(id:" << std::to_string(device_context->device_context_key().device_id_)
                    << ") memory isn't enough and alloc failed, kernel name: " << GetAID()
                    << ", alloc size: " + std::to_string(copy_device_tensor->GetSize()) << "B.";
      return true;
    }
    static std::string name = "Alloc memory";
    copy_kernel_tensor->IncreaseNewRefCount(name);
  }
  MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
    << "Alloc memory for device tensor:" << copy_device_tensor << " ptr:" << copy_device_tensor->GetPtr()
    << " size:" << copy_device_tensor->GetSize() << " index:" << i << " for actor:" << GetAID();
  if (type_ != KernelTransformType::kSuperKernelActor) {
    node_device_tensor->set_ptr(copy_device_tensor->GetMutablePtr());
  } else {
    node_device_tensor->set_ptr(copy_kernel_tensor->GetValidPtr(copy_device_tensor->stream_id()));
  }
  node_device_tensor->set_from_mem_pool(false);
  return false;
}

bool SuperKernelActor::CopyInputData(const OpContext<KernelTensor> *context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph);
  if (device_contexts_.empty() || device_contexts_[0] == nullptr ||
      device_contexts_[0]->device_res_manager_ == nullptr) {
    MS_LOG(ERROR) << "Invalid device context for actor:" << GetAID();
    return false;
  }
  auto device_context = device_contexts_[0];
  auto &input_nodes = graph->input_nodes();
  if (input_kernel_tensors_.size() != node_kernel_tensors_.size()) {
    MS_LOG(ERROR) << "The size of input_kernel_tensors_[" << input_kernel_tensors_.size()
                  << "] is not equal to the size of node_kernel_tensors_[" << node_kernel_tensors_.size() << "].";
    return false;
  }

  for (size_t i = 0; i < input_kernel_tensors_.size(); ++i) {
    auto &node_device_kernel_tensor = node_kernel_tensors_[i];
    MS_EXCEPTION_IF_NULL(node_device_kernel_tensor);
    auto &node_device_tensor = node_device_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(node_device_tensor);
    MS_EXCEPTION_IF_NULL(input_nodes[i]);
    auto &input_kernel_tensor = input_kernel_tensors_[i];
    if (InputDataNoNeedCopy(input_nodes[i], input_kernel_tensor, node_device_kernel_tensor, type_)) {
      MS_LOG(DEBUG) << "Actor:" << GetAID() << " input kernel tensor " << i << ":" << input_kernel_tensor
                    << " no need copy.";
      continue;
    }
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    UpdateShape(input_nodes[i], node_device_kernel_tensor, input_kernel_tensor, type_);
    node_device_kernel_tensor->set_user_data(input_kernel_tensors_[i]->user_data());
    node_device_kernel_tensor->set_need_sync_user_data(input_kernel_tensors_[i]->need_sync_user_data());
    if (type_ != KernelTransformType::kSuperKernelActor) {
      node_device_kernel_tensor->SetValue(input_kernel_tensor->GetValueTrack());
    }

    // Copy.
    KernelTensorPtr copy_kernel_tensor = nullptr;
    // If the input is not a persist device address, in a heterogeneous scenario, a new device address needs to
    // be created. And set ptr to node device address to support the zero copy of graph input nodes.
    if (!node_device_kernel_tensor->is_ptr_persisted()) {
      if (CopyInputDataPersistedHandle(device_context, input_kernel_tensors_[i], node_device_kernel_tensor, i)) {
        continue;
      }
      copy_kernel_tensor = copy_input_kernel_tensors_[i];
    } else {
      if (node_device_tensor->GetPtr() == nullptr) {
        MS_LOG(INFO) << "The node device tensor:" << node_device_tensor
                     << ", which shared with another graph, has no device memory and will skip "
                        "copy for actor:"
                     << GetAID();
        continue;
      }
      copy_kernel_tensor = node_device_kernel_tensor;
    }
    MS_EXCEPTION_IF_NULL(copy_kernel_tensor);
    MS_LOG(INFO) << "The input data of node:" << input_nodes[i]->DebugString()
                 << " need copy from kernel tensor:" << input_kernel_tensor->ToString()
                 << " to kernel tensor:" << copy_kernel_tensor->ToString()
                 << ", is ref node need copy back:" << is_parameters_need_copy_[i] << " for actor:" << GetAID();
    if (!SyncCopy(copy_kernel_tensor.get(), input_kernel_tensor.get(), kDefaultStreamIndex)) {
      MS_LOG(ERROR) << "Copy data failed for actor:" << GetAID() << " input index:" << i;
      continue;
    }

    if (is_parameters_need_copy_[i]) {
      ref_node_addr_map_[copy_kernel_tensor] = input_kernel_tensor;
    }
  }
  return true;
}

void SuperKernelActor::SendMemoryFreeReq(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);

  if (device_contexts_.empty() || device_contexts_[0] == nullptr ||
      device_contexts_[0]->device_res_manager_ == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                  "Invalid device context for super kernel actor:" + GetAID().Name());
  }

  if (memory_free_lists_.size() > 0 && memory_free_lists_.back().size() > 0) {
    if (memory::mem_pool::IsNeedProfilieMemoryLog()) {
      for (auto data : memory_free_lists_.back()) {
        auto output_address = data->device_address().get();
        MS_EXCEPTION_IF_NULL(output_address);
        MS_LOG(WARNING) << "Need Profile Memory, Memory need Decrease DynamicRefCount, actor name: " << GetAID().Name()
                        << ", kernel graph: " << graph_->ToString() << ", device address class ptr: " << output_address
                        << ", device address size: " << data->GetSize()
                        << ", device address addr: " << data->device_ptr();
      }
    }

    MS_LOG(DEBUG) << "Send memory free size:" << memory_free_lists_.back().size() << " for actor:" << GetAID();
    if (ActorDispatcher::is_memory_free_sync()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                                device_contexts_[0], context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                            device_contexts_[0], context, GetAID());
    }
  }

  // Free the address that is the temp store for kernel input copy.
  for (auto &copy_input_kernel_tensor : copy_input_kernel_tensors_) {
    if (copy_input_kernel_tensor == nullptr) {
      continue;
    }
    auto &copy_input_device_tensor = copy_input_kernel_tensor->device_address();
    if ((copy_input_device_tensor != nullptr) && (copy_input_device_tensor->GetPtr() != nullptr)) {
      device_contexts_[0]->device_res_manager_->FreeMemory(copy_input_device_tensor.get());
    }
  }
}

void SuperKernelActor::SetRelationForControlFlow() {
  MS_EXCEPTION_IF_NULL(graph_);
  if (graph_->inline_sub_graph_kernels().empty()) {
    return;
  }
  enable_inline_control_flow_ = true;
  std::map<std::string, KernelRunnerPtr> name_to_actors;
  for (const auto &kernel_actor : kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    name_to_actors[kernel_actor->GetAID().Name()] = kernel_actor;
  }
  const auto &gather_to_switch = graph_->condition_gather_to_switch();
  std::map<std::string, bool *> branch_name_to_flags;
  for (const auto &kernel_actor : kernel_actors_) {
    MS_EXCEPTION_IF_NULL(kernel_actor->kernel_);
    if (common::AnfAlgo::CheckPrimitiveType(kernel_actor->kernel_, prim::kPrimConditionGather)) {
      const auto &switch_node_iter = gather_to_switch.find(kernel_actor->kernel_);
      if (switch_node_iter == gather_to_switch.end()) {
        MS_LOG(EXCEPTION) << " Failed to get switch node for gather:" << kernel_actor->kernel_->fullname_with_scope()
                          << " in graph:" << graph_->ToString();
      }
      MS_EXCEPTION_IF_NULL(switch_node_iter->second);
      const auto &switch_actor_iter = name_to_actors.find(GetActorIdByKernel(switch_node_iter->second));
      if (switch_actor_iter == name_to_actors.end()) {
        MS_LOG(EXCEPTION) << " Failed to get switch actor for node:" << switch_node_iter->second->fullname_with_scope()
                          << " in graph:" << graph_->ToString();
      }
      const auto &actor = switch_actor_iter->second;
      MS_EXCEPTION_IF_NULL(actor);
      const auto &switch_actor = dynamic_cast<ConditionSwitchRunner *>(actor.get());
      MS_EXCEPTION_IF_NULL(switch_actor);
      const auto &gather_actor = dynamic_cast<ConditionGatherRunner *>(kernel_actor.get());
      MS_EXCEPTION_IF_NULL(gather_actor);
      switch_actor->gather_branch_name_ = &gather_actor->current_branch_name_;
      MS_EXCEPTION_IF_NULL(switch_actor->branch_flags_);
      gather_actor->branch_flags_ = switch_actor->branch_flags_;
    } else if (common::AnfAlgo::CheckPrimitiveType(kernel_actor->kernel_, prim::kPrimConditionSwitch)) {
      const auto &switch_actor = dynamic_cast<ConditionSwitchRunner *>(kernel_actor.get());
      MS_EXCEPTION_IF_NULL(switch_actor);
      std::shared_ptr<bool[]> flags(new bool[switch_actor->branch_names_.size()], std::default_delete<bool[]>());
      switch_actor->branch_flags_ = flags;
      MS_EXCEPTION_IF_NULL(switch_actor->branch_flags_);
      for (size_t i = 0; i < switch_actor->branch_names_.size(); ++i) {
        const auto &branch_name = switch_actor->branch_names_[i];
        branch_name_to_flags[branch_name] = &(switch_actor->branch_flags_.get()[i]);
        switch_actor->branch_flags_.get()[i] = false;
        MS_LOG(INFO) << "Add flag:" << branch_name_to_flags[branch_name] << " for branch:" << branch_name
                     << " in actor:" << kernel_actor->GetAID() << " graph:" << graph_->ToString();
      }
    }
    const auto &branch_name_iter = graph_->inline_sub_graph_kernels().find(kernel_actor->kernel_);
    if (branch_name_iter == graph_->inline_sub_graph_kernels().end()) {
      kernel_actor->is_enable_ = &enable_inline_control_flow_;
      continue;
    }
    auto branch_name = branch_name_iter->second;
    const auto &flag_iter = branch_name_to_flags.find(branch_name);
    if (flag_iter == branch_name_to_flags.end()) {
      MS_LOG(EXCEPTION) << "Failed to get branch flag by branch name:" << branch_name
                        << " node:" << kernel_actor->kernel_->fullname_with_scope()
                        << " in graph:" << graph_->ToString();
    }
    kernel_actor->is_enable_ = flag_iter->second;
    MS_LOG(DEBUG) << "Set flag:" << flag_iter->second << " for kernel actor:" << kernel_actor->GetAID()
                  << " graph:" << graph_->ToString();
  }
}

void SuperKernelActor::BuildAndLinkKernelActors() {
  MS_LOG(DEBUG) << "Build and link for actor:" << GetAID() << " kbk execute:" << enable_kbk_sub_graph_execute_;
  if (enable_kbk_sub_graph_execute_) {
    BuildKernelActors();
    LinkKernelActors();
    SetRelationForControlFlow();
    if (enable_parallel_dispatch_) {
      PartitionParallelDispatchKernels();
    }

    enable_capture_graph_ = GraphCaptureManager::GetInstance().GetEnableGraphCapture() &&
                            (graph_phase_.find("increment") != std::string::npos) && graph_->is_dynamic_shape();
    if (enable_capture_graph_) {
      auto ret =
        GraphCaptureManager::GetInstance().FindSupportCaptureKernelPositions(kernel_actors_, device_contexts_[0]);
      if (ret) {
        size_t stream_id = kIndex3;
        device_contexts_[0]->device_res_manager_->CreateStream(&stream_id);
        MS_LOG(WARNING) << "Create new stream for capture graph, stream id: " << stream_id;
        GraphCaptureManager::GetInstance().SetStreamId(stream_id);
        MS_LOG(INFO) << "Enable acl graph for graph: " << graph_->ToString() << ", phase: " << graph_phase_;
      } else {
        MS_LOG(INFO) << "Can't find any available op to capture in graph: " << graph_->ToString()
                     << ", phase: " << graph_phase_;
        GraphCaptureManager::GetInstance().SetEnableGraphCapture(false);
        enable_capture_graph_ = false;
      }
    }
  }
}

void SuperKernelActor::PartitionParallelDispatchKernels() {
  auto runtime_conf_instance = RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  parallel_dispatch_num_ = runtime_conf_instance->group_launch_thread_num();
  if (parallel_dispatch_num_ < 1) {
    MS_LOG(EXCEPTION) << "Invalid thread num: " << parallel_dispatch_num_
                      << " for kernel launch group, please check the `thread_num` value of function: "
                         "runtime.set_kernel_launch_group(thread_num, kernel_group_num)";
  }
  MS_LOG(INFO) << "The parallel dispatch thread number: " << parallel_dispatch_num_;

  auto total_kernel_group_num = runtime_conf_instance->kernel_group_num();
  parallel_slice_num_ = total_kernel_group_num / parallel_dispatch_num_;
  if (parallel_slice_num_ < 1) {
    MS_LOG(EXCEPTION) << "Invalid kernel group num: " << total_kernel_group_num
                      << ", kernel group num must be greater than or equal to thread num: " << parallel_dispatch_num_
                      << ", please check the parameter value of function: "
                         "runtime.set_kernel_launch_group(thread_num, kernel_group_num)";
  }
  MS_LOG(INFO) << "The kernel group per thread: " << parallel_slice_num_;

  // Get parallel launch kernels slice/group.
  parallel_launch_kernels_.resize(parallel_dispatch_num_ * parallel_slice_num_);
  size_t total_kernel_num = kernel_actors_.size();
  size_t kernel_num_per_dispatcher = total_kernel_num / (parallel_dispatch_num_ * parallel_slice_num_);
  MS_LOG(INFO) << "Total kernel num: " << kernel_actors_.size();
  MS_LOG(INFO) << "The kernel num per parallel slice: " << kernel_num_per_dispatcher;
  auto begin_iter = kernel_actors_.begin();
  for (size_t i = 0; i < parallel_launch_kernels_.size(); i++) {
    if (i < parallel_launch_kernels_.size() - 1) {
      parallel_launch_kernels_[i] = std::vector<KernelRunnerPtr>(begin_iter + i * kernel_num_per_dispatcher,
                                                                 begin_iter + (i + 1) * kernel_num_per_dispatcher);
    } else {
      parallel_launch_kernels_[i] =
        std::vector<KernelRunnerPtr>(begin_iter + i * kernel_num_per_dispatcher, kernel_actors_.end());
    }
    MS_LOG(INFO) << "The kernel group[" << i << "] kernel num: " << parallel_launch_kernels_[i].size();
  }

  // Get serial launch kernels.
  static bool enable_multi_comm_group = runtime::IsEnableRuntimeConfig(runtime::kRuntimeCommunicationLaunchGroup);
  for (auto &kernel_actor : kernel_actors_) {
    if (!kernel_actor) {
      continue;
    }
    auto &llm_manager = LLMManager::GetInstance();
    const auto &kernel_name = kernel_actor->kernel_mod_->kernel_name();
    bool need_force_resize = llm_manager.need_force_resize(kernel_name) || kernel_actor->is_dynamic_value_;
    // The heterogeneous operator like MoveTo, device context type maybe 'Ascend' type, but output DeviceAddress type is
    // 'CPU'.
    bool is_heter_op = (kernel_actor->real_output_device_context_ != kernel_actor->device_contexts_[0]);
    if (need_force_resize || common::AnfAlgo::IsCommunicationFusionOp(kernel_name) || is_heter_op) {
      serial_launch_kernels_.push_back(kernel_actor);
      continue;
    }
    if (common::AnfAlgo::IsCommunicationOp(kernel_actor->kernel_)) {
      if (!enable_multi_comm_group) {
        serial_launch_kernels_.push_back(kernel_actor);
      }
      continue;
    }

    if (kernel_name.find(kAllReduceOpName) != std::string::npos ||
        kernel_name.find(kAllGatherOpName) != std::string::npos ||
        kernel_name.find(kReduceScatterOpName) != std::string::npos ||
        kernel_name.find(kAllToAllOpName) != std::string::npos ||
        kernel_name.find(kAlltoAllOpName) != std::string::npos) {
      MS_LOG(WARNING) << "Find not support parallel launch communication op: " << kernel_name;
      serial_launch_kernels_.push_back(kernel_actor);
    }
  }

  if (enable_multi_comm_group) {
    RecreateCommunicationGroup();
  }
}

void SuperKernelActor::RecreateCommunicationGroup() {
  std::vector<std::vector<KernelRunnerPtr>> parallel_launch_comm_kernels(parallel_dispatch_num_);
  HashSet<std::string> group_set;
  // 1. Collect communication ops.
  for (size_t i = 0; i < parallel_dispatch_num_; i++) {
    for (size_t j = 0; j < parallel_slice_num_; j++) {
      auto &kernel_actors = parallel_launch_kernels_[i + j * parallel_dispatch_num_];
      for (auto &kernel_actor : kernel_actors) {
        if (!kernel_actor) {
          continue;
        }

        const auto &kernel_name = kernel_actor->kernel_mod_->kernel_name();
        // MC2 kernels do not support multi communication group now.
        bool is_naive_comm_op = common::AnfAlgo::IsNaiveCommunicationOp(kernel_name);
        if (!is_naive_comm_op) {
          continue;
        }

        parallel_launch_comm_kernels[i].push_back(kernel_actor);
        if (common::AnfAlgo::HasNodeAttr(kAttrGroup, kernel_actor->kernel_)) {
          auto group_name = common::AnfAlgo::GetNodeAttr<std::string>(kernel_actor->kernel_, kAttrGroup);
          group_set.insert(group_name);
        } else {
          MS_LOG(EXCEPTION) << "Can not get communication group for kernel: "
                            << kernel_actor->kernel_->fullname_with_scope();
        }
      }
    }
  }

  if (group_set.size() > 1) {
    MS_LOG(EXCEPTION)
      << "Communication ops parallel dispatch doesn't support multi communication group now, please disable "
         "parallel dispatch for communication ops by: export MS_DEV_RUNTIME_CONF='communication_launch_group:False'";
  }
  std::string old_group_name = "";
  std::vector<uint32_t> group_ranks = {};
  if (group_set.size() == 1) {
    old_group_name = *group_set.begin();
    group_ranks = distributed::collective::CollectiveManager::instance()->GetGroupRanks(old_group_name);
    MS_LOG(INFO) << "Old group name: " << old_group_name << ", group ranks: " << group_ranks;
  } else {
    MS_LOG(WARNING) << "There is no communication ops can parallel launch.";
    return;
  }

  MS_LOG(INFO) << "Enable parallel launch communication ops.";
  for (size_t i = 0; i < parallel_launch_comm_kernels.size(); i++) {
    auto &comm_kernel_actors = parallel_launch_comm_kernels[i];
    if (comm_kernel_actors.empty()) {
      continue;
    }
    // 2. New communication group.
    const std::string new_group_name = std::string("parallel_dispatch_group_") + std::to_string(i);
    distributed::collective::CollectiveManager::instance()->CreateCommunicationGroup(new_group_name, group_ranks);

    // 3. Repalce old communication group and re-init kernel mod for communication ops.
    for (auto &kernel_actor : comm_kernel_actors) {
      auto &kernel = kernel_actor->kernel_;
      MS_EXCEPTION_IF_NULL(kernel_actor);
      common::AnfAlgo::SetNodeAttr(kAttrGroup, MakeValue<std::string>(new_group_name), kernel);

      std::vector<KernelTensor *> input_kernel_tensors = AnfAlgo::GetOrCreateAllInputKernelTensors(kernel);
      std::vector<KernelTensor *> output_kernel_tensors = AnfAlgo::GetOrCreateAllOutputKernelTensors(kernel);

      MS_LOG(INFO) << "Begin init kernel: " << kernel->fullname_with_scope();
      if (!kernel_actor->kernel_mod_->Init(common::AnfAlgo::GetCNodePrimitive(kernel), input_kernel_tensors,
                                           output_kernel_tensors)) {
        MS_LOG_WITH_NODE(EXCEPTION, kernel)
          << "#dmsg#Kernel build failed:#dmsg#Initialize kernel op[" << kernel->fullname_with_scope() << "] failed.";
      }
      MS_LOG(INFO) << "End init kernel: " << kernel->fullname_with_scope();

      if (kernel::CheckResizeCondition(kernel)) {
        MS_LOG(INFO) << "Begin Resize kernel: " << kernel->fullname_with_scope();
        kernel_actor->kernel_mod_->Resize(input_kernel_tensors, output_kernel_tensors);
        MS_LOG(INFO) << "End Resize kernel: " << kernel->fullname_with_scope();
      }
    }
  }
}

KernelRunnerPtr SuperKernelActor::BuildInnerControlFlowActor(const CNodePtr &kernel,
                                                             const DeviceContext *device_context,
                                                             GraphExecutionStrategy strategy,
                                                             const std::set<size_t> &ref_input_indexes,
                                                             const std::set<size_t> &ref_output_indexes) {
  MS_EXCEPTION_IF_NULL(kernel);
  if (!common::AnfAlgo::CheckPrimitiveType(kernel, prim::kPrimConditionGather) &&
      !common::AnfAlgo::CheckPrimitiveType(kernel, prim::kPrimConditionSwitch)) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, kernel)
      << "#dmsg#Runtime error info:#dmsg#Kernel " << kernel->fullname_with_scope()
      << " is not a inner control flow kernel.";
  }
  if (common::AnfAlgo::CheckPrimitiveType(kernel, prim::kPrimConditionSwitch)) {
    return std::make_shared<ConditionSwitchRunner>(GenerateActorIdByKernel(kernel), kernel, device_context,
                                                   memory_manager_aid_, debug_aid_, recorder_aid_, strategy,
                                                   ref_input_indexes, ref_output_indexes);
  }
  return std::make_shared<ConditionGatherRunner>(GenerateActorIdByKernel(kernel), kernel, device_context,
                                                 memory_manager_aid_, debug_aid_, recorder_aid_, strategy,
                                                 ref_input_indexes, ref_output_indexes);
}

void SuperKernelActor::BuildKernelActors() {
  // 1. Generate KernelRunner for each Kernel in graph_ if need.
  GenerateKernelRunners();

  // 2. Add somas info.
  MS_EXCEPTION_IF_NULL(graph_);
  for (const auto &front_backend_pair : graph_->front_node_to_graph_output_map()) {
    const auto &output_with_index = front_backend_pair.second;
    auto output_kernel = output_with_index.first;
    auto output_index = output_with_index.second;
    MS_EXCEPTION_IF_NULL(output_kernel);
    auto origin_output_with_index = front_backend_pair.first;
    if (origin_output_with_index.first == nullptr) {
      MS_LOG(WARNING) << "The graph " << graph_->graph_id() << " output node:" << output_kernel->fullname_with_scope()
                      << " with index: " << output_index << " has no front node.";
      continue;
    }
    if (!AnfUtils::IsRealCNodeKernel(output_kernel)) {
      auto real_output_pair = common::AnfAlgo::FetchRealNodeSkipMonadControl({output_kernel, 0});
      if (!AnfUtils::IsRealCNodeKernel(real_output_pair.first)) {
        continue;
      }
      output_kernel = real_output_pair.first;
      output_index = real_output_pair.second;
    }
    auto iter = cnode_to_kernel_actor_.find(output_kernel);
    if (iter == cnode_to_kernel_actor_.end()) {
      MS_LOG_WITH_NODE(EXCEPTION, output_kernel)
        << "Can not find kernel actor for node: " << output_kernel->fullname_with_scope();
    }
    const auto &output_actor = iter->second;
    MS_EXCEPTION_IF_NULL(output_actor);
    SchedulerHelper::AddSomasInfoForGraphOutputV2(output_actor, output_index, graph_->graph_id());
  }

  // 3. Set free index for input and output address.
  // this step must execute before step 4 to get the input and output free index, so that the Init of kernel actor
  // could get the device address to be free.
  SetFreePositionForKernelActor();

  const auto &outputs = common::AnfAlgo::GetAllOutputWithIndex(graph_->output());
  for (const auto &output : outputs) {
    MS_EXCEPTION_IF_NULL(output.first);
    const auto &iter = cnode_to_kernel_actor_.find(output.first);
    if (iter == cnode_to_kernel_actor_.end()) {
      continue;
    }
    const auto &actor = iter->second;
    MS_EXCEPTION_IF_NULL(actor);
    if (output.second >= actor->is_output_kernel_.size()) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << output.second << " size:" << actor->is_output_kernel_.size()
                        << " for actor:" << actor->GetAID();
    }
    actor->is_output_kernel_[output.second] = true;
  }

  // 4. Initialize all kernel actor.
  // Note: this step must execute before LinkKernelActors, LinkKernelActors will check whether the output ref count is
  // max or not to optimize free performance for somas case, need not try to free the output which has a max ref
  // count.
  size_t kernel_num = kernel_actors_.size();
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel_actor = kernel_actors_[i];
    if (kernel_actor) {
      kernel_actor->Init();
    }
  }
}

void SuperKernelActor::GenerateKernelRunners() {
  MS_EXCEPTION_IF_NULL(graph_);
  const auto &execution_order = graph_->execution_order();
  size_t kernel_num = execution_order.size();
  MS_VLOG(VL_RUNTIME_FRAMEWORK_PRINT_PROF) << "Build " << kernel_num << " kernels for SuperKernelActor.";
  kernel_actors_.resize(kernel_num);

  mindspore::HashMap<uint32_t, std::pair<KernelRunnerPtr, KernelRunnerPtr>> send_recv_nodes;
  // 1. Create kernel actor if need.
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel = execution_order[i];
    MS_EXCEPTION_IF_NULL(kernel);
    if (IsSkippedKernelActor(kernel)) {
      kernel_actors_[i] = nullptr;
      continue;
    }

    if (!IsKernelActor(kernel, GraphExecutionStrategy::kPipeline)) {
      MS_LOG(WARNING) << "Find not real cnode in execution order for graph: " << graph_->graph_id();
      kernel_actors_[i] = nullptr;
      continue;
    }

    auto ref_input_indexes = FetchModifiableRefInputIndex(kernel);
    auto ref_output_indexes = FetchModifiableRefOutputIndex(kernel, graph_);
    auto real_device_context =
      const_cast<device::DeviceContext *>(device::FetchRealDeviceContext(kernel, device_contexts_[0]));
    MS_EXCEPTION_IF_NULL(real_device_context);
    KernelAsyncLaunchActor::GetInstance()->AddDeviceContext(real_device_context);
    RuntimePipeline::GetInstance().AddDeviceContext(real_device_context);

    if (enable_parallel_dispatch_ && (real_device_context->GetDeviceType() != device_contexts_[0]->GetDeviceType())) {
      // CPU kernels doesn't has stream context and can not parallel launch;
      MS_LOG(EXCEPTION)
        << "Find heterogeneous kernel: " << kernel->fullname_with_scope() << " in kernel graph: " << graph_->ToString()
        << ". The kernel group launch(parallel launch) feature can not work in this case. Please eliminate "
           "heterogeneous(cpu) kernel or disable kernel group launch feature.";
    }

    if (IsInnerControlFlowActor(kernel)) {
      kernel_actors_[i] = BuildInnerControlFlowActor(kernel, real_device_context, GraphExecutionStrategy::kPipeline,
                                                     ref_input_indexes, ref_output_indexes);
      SchedulerHelper::AddSomasInfoV2(kernel_actors_[i].get());
      cnode_to_kernel_actor_[kernel] = kernel_actors_[i].get();
      continue;
    }

    KernelRunnerPtr kernel_actor = std::make_shared<KernelRunner>(
      GenerateActorIdByKernel(kernel), kernel, real_device_context, memory_manager_aid_, debug_aid_, recorder_aid_,
      GraphExecutionStrategy::kPipeline, ref_input_indexes, ref_output_indexes);
    MS_EXCEPTION_IF_NULL(kernel_actor);
    kernel_actors_[i] = kernel_actor;

    // Set the member of kernel actor.
    kernel_actor->is_launch_skipped_ =
      common::AnfAlgo::IsNopNode(kernel) && graph_->IsInRefOutputMap(std::make_pair(kernel, 0));
    kernel_actor->inputs_continuous_memory_ =
      AnfAlgo::IsNeedContinuesMemoryOp(kernel) && (common::AnfAlgo::GetInputTensorNum(kernel) > 1);

    if (SchedulerHelper::IsSkipLaunchShapeRelatedOpV2(kernel_actor.get())) {
      kernel_actor->set_skip_launch_shape_related_op(true);
    }

    if (IsPrimitiveCNode(kernel, prim::kPrimStreamSend)) {
      SchedulerHelper::ProcessStreamSendRecvEventPairV2(&send_recv_nodes, kernel, kernel_actor, true);
    } else if (IsPrimitiveCNode(kernel, prim::kPrimStreamRecv)) {
      SchedulerHelper::ProcessStreamSendRecvEventPairV2(&send_recv_nodes, kernel, kernel_actor, false);
    }

    SchedulerHelper::AddSomasInfoV2(kernel_actor.get());

    cnode_to_kernel_actor_[kernel] = kernel_actor.get();
  }
  for (auto &[event_pair_id, send_recv_actor] : send_recv_nodes) {
    auto [send_actor, recv_actor] = send_recv_actor;
    MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Stream send/recv pair : " << event_pair_id
                                         << ", send_actor : " << send_actor << ", recv_actor : " << recv_actor << ".";
    recv_actor->set_stream_send_actor(send_actor.get());
  }
}

SuperKernelActor::SuperKernelActor(const std::string &name, const KernelGraphPtr &graph, const std::string &graph_phase,
                                   const DeviceContext *device_context, const AID &memory_manager_aid,
                                   const AID *debug_aid, const AID *recorder_aid, KernelTransformType type)
    : DebugAwareActor(name, type, recorder_aid, memory_manager_aid, debug_aid, nullptr),
      graph_(graph),
      graph_phase_(graph_phase),
      is_infer_phase_(IsInferPhase(graph_phase)),
      enable_kbk_sub_graph_execute_(EnableKbkSubGraphExecute()) {
  (void)device_contexts_.emplace_back(device_context);
  input_kernel_tensors_.resize(graph->input_nodes().size());
  std::vector<bool> is_enable_inputs(graph->input_nodes().size(), true);
  is_input_used_.swap(is_enable_inputs);
  kernel_async_infer_aid_ = KernelAsyncInferActor::GetInstance()->GetAID();
  kernel_async_resize_aid_ = KernelAsyncResizeActor::GetInstance()->GetAID();
  kernel_async_launch_aid_ = KernelAsyncLaunchActor::GetInstance()->GetAID();
  somas_info_ = graph_->MutableSomasInfo();

  enable_trace_memory_ = EnableTraceMemory();
  enable_parallel_dispatch_ = EnableParallelDispatchKernel() && MsContext::GetInstance()->IsEnableInferBoost() &&
                              (graph_phase_.find("increment") != std::string::npos) && enable_trace_memory_;
  MS_LOG(INFO) << "The kernel graph: " << graph_->ToString() << " phase: " << graph_phase_
               << ", enable parallel dispatch kernel: " << enable_parallel_dispatch_;

  is_high_perf_mode_ = IsHighPerfModeAtComp();
  std::for_each(profiler::Profiler::GetInstanceMap().begin(), profiler::Profiler::GetInstanceMap().end(),
                [this](const auto &prof_inst) { prof_instances_.emplace_back(prof_inst.second); });
}

bool SuperKernelActor::IsHighPerfModeAtComp() {
  // These high performance checking flag should be confirmed in compilation phase.
  // They should not be changed in runtime phase so that we could reach optimal performance by avoiding condition
  // judgement.
  std::vector<bool> conditions = {
    runtime::IsDisableRuntimeConfig(runtime::kRuntimeHPMode),
    debug_aid_ != nullptr,
    recorder_aid_ != nullptr,
    EnableExecuteOrderDump(),
    device::tracker::MemTrackerManager::GetInstance().IsEnabled(),
    tools::TftConfig::GetInstance()->IsEnableUCE(),
    mindspore::runtime::RuntimeConf::GetInstance()->launch_blocking(),
    common::GetEnv("MS_ENABLE_CKPT_D2H_ASYNC") == "1",
    memory::mem_pool::IsNeedProfilieMemoryLog(),
    tools::TftConfig::IsEnableStepTRE(),
    tools::TftConfig::GetInstance() != nullptr && tools::TftConfig::GetInstance()->IsEnableSaveHcclOpStatus(),
  };
  // When this function returns false, it means performance is not cirtical in this context.
  // Otherwise runtime will launch kernels with high performance.
  bool is_hp_at_comp = std::all_of(conditions.begin(), conditions.end(), [](bool c) { return !c; });
  MS_LOG(INFO) << "Whether this is high performance mode at compilation phase: " << is_hp_at_comp;
  return is_hp_at_comp;
}

bool SuperKernelActor::IsHighPerfModeAtExec() {
  // These flags will be switched during execution.
  // For each step, SuperKernelActor should check this function's return value.
  std::vector<bool> conditions = {
    device::tracker::MemTrackerManager::GetInstance().enable_memory_debug_info(),
    mindspore::runtime::ProfilerAnalyzer::GetInstance().profiler_enable(),
    std::any_of(prof_instances_.begin(), prof_instances_.end(),
                [](const auto &p) { return p->GetEnableFlag() || p->GetOpTimeFlag(); }),
  };

  bool is_hp_at_exec = std::all_of(conditions.begin(), conditions.end(), [](bool c) { return !c; });
  MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL) << "Whether this is high performance mode at execution phase: " << is_hp_at_exec;
  return is_hp_at_exec;
}

void SuperKernelActor::ResetState(OpContext<KernelTensor> *const context) {
  if ((somas_info_ != nullptr) && (somas_info_->whole_block_size_ != 0)) {
    MemoryManagerActor::GetInstance()->FreeSomasMemory(somas_info_, device_contexts_[0], context, GetAID());
  }
}

void SuperKernelActor::GetRefCountForGraphOutput(const std::vector<AnfNodePtr> &output_data_nodes,
                                                 const std::vector<DataArrowPtr> &output_data_arrows,
                                                 const mindspore::HashMap<AnfNodePtr, KernelRunner *> &kernel_to_actor,
                                                 const std::map<uint32_t, std::vector<CNodePtr>> &inplace_groups,
                                                 const std::string &actor_name) {
  mindspore::HashMap<KernelRunner *, mindspore::HashMap<size_t, size_t>> kernel_actor_to_increase_new_ref_count;
  if (output_data_nodes.size() != output_data_arrows.size()) {
    MS_LOG(EXCEPTION) << "Invalid output data node size:" << output_data_nodes.size()
                      << " and arrow size:" << output_data_arrows.size() << " for actor:" << actor_name;
  }
  for (size_t i = 0; i < output_data_nodes.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_data_nodes[i]);
    MS_EXCEPTION_IF_NULL(output_data_arrows[i]);
    const auto &real_node_with_index = common::AnfAlgo::VisitKernelWithReturnType(
      output_data_nodes[i], output_data_arrows[i]->from_output_index_, false);
    MS_EXCEPTION_IF_NULL(real_node_with_index.first);
    MS_LOG(DEBUG) << "Check output node:" << output_data_nodes[i]->fullname_with_scope()
                  << " real node:" << real_node_with_index.first->fullname_with_scope()
                  << " index:" << real_node_with_index.second << " to actor:" << output_data_arrows[i]->to_op_id_
                  << " to index:" << output_data_arrows[i]->to_input_index_ << " for actor:" << actor_name;
    if (real_node_with_index.first->isa<CNode>()) {
      if (!AnfAlgo::OutputAddrExist(real_node_with_index.first, real_node_with_index.second, false)) {
        MS_LOG(EXCEPTION) << "Failed to get output device address in node:"
                          << real_node_with_index.first->fullname_with_scope()
                          << " index:" << real_node_with_index.second << " for actor:" << actor_name;
      }
      const auto &device_tensor =
        AnfAlgo::GetMutableOutputAddr(real_node_with_index.first, real_node_with_index.second, false).get();
      MS_EXCEPTION_IF_NULL(device_tensor);
      auto actor_iter = kernel_to_actor.find(real_node_with_index.first);
      if (actor_iter == kernel_to_actor.end()) {
        MS_LOG(EXCEPTION) << "Failed to get actor by kernel:" << real_node_with_index.first->fullname_with_scope()
                          << " debug string:" << real_node_with_index.first->DebugString()
                          << " in graph:" << graph_->ToString() << " for actor:" << GetAID();
      }
      MS_EXCEPTION_IF_NULL(actor_iter->second);
      actor_iter->second->increase_ref_count_size_[real_node_with_index.second]++;
      kernel_actor_to_increase_new_ref_count[actor_iter->second][real_node_with_index.second]++;
    }
  }

  for (const auto &group : inplace_groups) {
    if (group.second.size() <= 1) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(group.second[0]);
    auto actor_iter = kernel_to_actor.find(group.second[0]);
    if (actor_iter == kernel_to_actor.end()) {
      MS_LOG(EXCEPTION) << "Failed to get actor by kernel:" << group.second[0]->fullname_with_scope();
    }
    MS_EXCEPTION_IF_NULL(actor_iter->second);
    actor_iter->second->increase_ref_count_size_[0] += group.second.size() - 1;
    kernel_actor_to_increase_new_ref_count[actor_iter->second][0] += group.second.size() - 1;
    MS_LOG(DEBUG) << "Add new ref count:" << (group.second.size() - 1)
                  << " for inplace group first node:" << group.second[0]->fullname_with_scope();
  }

  for (const auto &pair : kernel_actor_to_increase_new_ref_count) {
    for (const auto &sub_pair : pair.second) {
      MS_LOG(DEBUG) << "Actor:" << pair.first->GetAID() << " output index:" << sub_pair.first
                    << " should add new ref count size:" << sub_pair.second;
    }
  }
}

std::string GetBranchNameByIndex(const KernelRunnerPtr &kernel_actor, const AnfNodePtr &input_node,
                                 size_t input_index) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  MS_EXCEPTION_IF_NULL(kernel_actor->kernel());
  if (!common::AnfAlgo::CheckPrimitiveType(kernel_actor->kernel(), prim::kPrimConditionGather)) {
    MS_LOG(EXCEPTION) << "Invalid gather actor:" << kernel_actor->GetAID();
  }
  if (!kernel_actor->kernel()->HasAttr(kAttrBranchOutputNum)) {
    MS_LOG(EXCEPTION) << "Failed to get branch output num by condition gather actor:"
                      << kernel_actor->kernel()->fullname_with_scope()
                      << " input node:" << input_node->fullname_with_scope() << " in actor:" << kernel_actor->GetAID();
  }
  const auto &output_value = kernel_actor->kernel()->GetAttr(kAttrBranchOutputNum);
  MS_EXCEPTION_IF_NULL(output_value);
  size_t branch_output_num = GetValue<size_t>(output_value);
  if (!kernel_actor->kernel()->HasAttr(kAttrBranchGraphName)) {
    MS_LOG(EXCEPTION) << "Failed to get inline graph name by condition gather actor:"
                      << kernel_actor->kernel()->fullname_with_scope()
                      << " input node:" << input_node->fullname_with_scope() << " in actor:" << kernel_actor->GetAID();
  }
  const auto &branch_graph_names = kernel_actor->kernel()->GetAttr(kAttrBranchGraphName);
  MS_EXCEPTION_IF_NULL(branch_graph_names);
  MS_LOG(DEBUG) << "Branch graph name:" << branch_graph_names->ToString() << " for actor:" << kernel_actor->GetAID();
  if (!branch_graph_names->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Invalid branch group name:" << branch_graph_names->ToString()
                      << " for gather actor:" << kernel_actor->kernel()->fullname_with_scope()
                      << " input node:" << input_node->fullname_with_scope() << " in actor:" << kernel_actor->GetAID();
  }
  const auto &tuple_name = branch_graph_names->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_name);
  if (input_index / branch_output_num >= tuple_name->size()) {
    MS_LOG(EXCEPTION) << "Invalid input index:" << input_index
                      << " for input node:" << input_node->fullname_with_scope()
                      << " branch output size:" << branch_output_num
                      << " branch name:" << branch_graph_names->ToString()
                      << " for gather actor:" << kernel_actor->kernel()->fullname_with_scope();
  }
  return GetValue<std::string>(tuple_name->value()[input_index / branch_output_num]);
}

void SuperKernelActor::SetInputFreePositionForKernelActor(
  const KernelRunnerPtr &kernel_actor,
  const mindspore::HashMap<AnfNodePtr, device::DeviceContextKey> &kernel_to_context_key,
  const device::DeviceContextKey &graph_device_context_key,
  std::set<std::pair<KernelWithIndex, FreeNodeInfo>> *checked_nodes) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  MS_EXCEPTION_IF_NULL(checked_nodes);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_actor->kernel_);
  for (size_t i = 0; i < input_num; ++i) {
    if (i < kernel_actor->depend_shape_input_list_.size() && kernel_actor->depend_shape_input_list_[i]) {
      MS_LOG(DEBUG) << "Actor:" << kernel_actor->GetAID() << " skip check free input device tensor index:" << i;
      continue;
    }
    auto input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(kernel_actor->kernel_, i, false);
    if (IsSkippedKernelActor(input_node_with_index.first)) {
      input_node_with_index = common::AnfAlgo::GetPrevNodeOutput(input_node_with_index.first, 0, false);
      if (input_node_with_index.first == nullptr || !input_node_with_index.first->isa<CNode>()) {
        MS_INTERNAL_EXCEPTION(RuntimeError)
          << "Invalid skip kernel input " << i << " for kernel:" << kernel_actor->kernel_->fullname_with_scope()
          << " in super kernel actor:" << GetAID();
      }
      MS_LOG(DEBUG) << "Skip input node:" << input_node_with_index.first->fullname_with_scope()
                    << " index:" << input_node_with_index.second
                    << " for kernel:" << kernel_actor->kernel_->fullname_with_scope()
                    << " in super kernel actor:" << GetAID();
    }
    const auto &real_input_node_with_index =
      common::AnfAlgo::VisitKernelWithReturnType(input_node_with_index.first, input_node_with_index.second, false);
    const auto &input_node = real_input_node_with_index.first;
    MS_EXCEPTION_IF_NULL(input_node);
    FreeNodeInfo input_info = {graph_device_context_key, ""};
    if (input_node->isa<CNode>()) {
      const auto &output_context_iter = kernel_to_context_key.find(kernel_actor->kernel_);
      if (output_context_iter != kernel_to_context_key.end()) {
        input_info.context_key = output_context_iter->second;
        if (SchedulerHelper::IsIgnoredInputAddressV2(kernel_actor.get(), i)) {
          const auto &input_context_iter = kernel_to_context_key.find(input_node);
          if (input_context_iter != kernel_to_context_key.end()) {
            input_info.context_key = input_context_iter->second;
            MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL)
              << "Heter input kernel:" << input_node->fullname_with_scope()
              << " for kernel actor:" << kernel_actor->GetAID() << " in actor:" << GetAID();
          } else {
            MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL)
              << "Failed to get device context key for input node:" << input_node->DebugString()
              << " of kernel:" << kernel_actor->kernel_->fullname_with_scope() << " in actor:" << GetAID();
          }
        }
      } else {
        MS_LOG(WARNING) << "Failed to get device context key for input node:" << input_node->DebugString()
                        << " of kernel:" << kernel_actor->kernel_->fullname_with_scope() << " in actor:" << GetAID();
      }
      if (common::AnfAlgo::CheckPrimitiveType(input_node, prim::kPrimConditionSwitch)) {
        const auto &iter = graph_->inline_sub_graph_kernels().find(kernel_actor->kernel_);
        if (iter == graph_->inline_sub_graph_kernels().end()) {
          if (!common::AnfAlgo::CheckPrimitiveType(kernel_actor->kernel_, prim::kPrimConditionGather)) {
            MS_LOG(EXCEPTION) << "Failed to get branch info for kernel:" << kernel_actor->kernel_->fullname_with_scope()
                              << " input node:" << input_node->fullname_with_scope() << " in actor:" << GetAID();
          }
          input_info.branch_name = GetBranchNameByIndex(kernel_actor, input_node, i);
          MS_LOG(INFO) << "Input branch name:" << input_info.branch_name << " for input index:" << i
                       << " input node:" << input_node->fullname_with_scope()
                       << " for gather actor:" << kernel_actor->kernel_->fullname_with_scope();
        } else {
          input_info.branch_name = iter->second;
        }
      }
    }

    if (checked_nodes->find({real_input_node_with_index, input_info}) != checked_nodes->end() ||
        input_node->isa<ValueNode>()) {
      continue;
    }
    checked_nodes->emplace(real_input_node_with_index, input_info);
    MS_LOG(DEBUG) << "Get real input node:" << real_input_node_with_index.first->DebugString()
                  << " context key:" << input_info.context_key.ToString()
                  << " for kernel:" << kernel_actor->kernel_->fullname_with_scope();
    if (input_node->isa<Parameter>()) {
      auto iter = std::find(graph_->input_nodes().begin(), graph_->input_nodes().end(), input_node);
      if (iter == graph_->input_nodes().end()) {
        MS_LOG(EXCEPTION) << "Failed to find parameter:" << input_node->DebugString()
                          << " in graph:" << graph_->ToString();
      }
      size_t input_position = LongToSize(iter - graph_->input_nodes().begin());
      is_input_used_[input_position] = true;
    }
    kernel_actor->input_free_index_.emplace_back(i);
    MS_LOG(DEBUG) << "Add free input index:" << i << " for actor:" << kernel_actor->GetAID();
  }
}

void SuperKernelActor::SetOutputFreePositionForKernelActor(
  const KernelRunnerPtr &kernel_actor,
  const mindspore::HashMap<AnfNodePtr, device::DeviceContextKey> &kernel_to_context_key,
  const device::DeviceContextKey &graph_device_context_key,
  std::set<std::pair<KernelWithIndex, FreeNodeInfo>> *checked_nodes) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  MS_EXCEPTION_IF_NULL(checked_nodes);
  const auto kernel_info = dynamic_cast<KernelInfo *>(kernel_actor->kernel_->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info);
  size_t output_num = kernel_info->output_kernel_tensor_list().size();
  for (size_t i = 0; i < output_num; ++i) {
    const auto &real_node_with_index = common::AnfAlgo::VisitKernelWithReturnType(kernel_actor->kernel_, i, false);
    FreeNodeInfo output_info = {graph_device_context_key, ""};
    const auto &output_context_iter = kernel_to_context_key.find(kernel_actor->kernel_);
    if (output_context_iter != kernel_to_context_key.end()) {
      output_info.context_key = output_context_iter->second;
    } else {
      MS_LOG(WARNING) << "Failed to get device context key for kernel:" << kernel_actor->kernel_->fullname_with_scope()
                      << " in actor:" << GetAID();
    }
    if (!common::AnfAlgo::CheckPrimitiveType(kernel_actor->kernel_, prim::kPrimConditionSwitch)) {
      if (checked_nodes->find({real_node_with_index, output_info}) != checked_nodes->end()) {
        continue;
      }
      checked_nodes->emplace(real_node_with_index, output_info);
      kernel_actor->output_free_index_.emplace_back(i);
      MS_LOG(DEBUG) << "Add free output index:" << i << " context key:" << output_info.context_key.ToString()
                    << " for actor:" << kernel_actor->GetAID();
      continue;
    }
    const auto &switch_actor = dynamic_cast<ConditionSwitchRunner *>(kernel_actor.get());
    MS_EXCEPTION_IF_NULL(switch_actor);
    if (!switch_actor->kernel_->HasAttr(kInlineSubGraphName)) {
      MS_LOG(EXCEPTION) << "Failed to get branch name by actor:" << switch_actor->GetAID();
    }
    const auto &branch_name_value = switch_actor->kernel_->GetAttr(kInlineSubGraphName);
    MS_EXCEPTION_IF_NULL(branch_name_value);
    MS_LOG(DEBUG) << "inline branch name:" << branch_name_value->ToString() << " for actor:" << GetAID();
    if (!branch_name_value->isa<ValueTuple>()) {
      MS_LOG(EXCEPTION) << "Invalid branch name:" << branch_name_value->ToString() << " for actor:" << GetAID();
    }
    const auto &tuple_name = branch_name_value->cast<ValueTuplePtr>();
    MS_EXCEPTION_IF_NULL(tuple_name);
    MS_LOG(DEBUG) << "Check output free position for condition switch actor:" << switch_actor->GetAID()
                  << " branch:" << switch_actor->branch_names_;
    for (const auto &name_value : tuple_name->value()) {
      output_info.branch_name = GetValue<std::string>(name_value);
      MS_LOG(DEBUG) << "Check branch:" << output_info.branch_name << " for actor:" << switch_actor->GetAID();
      if (checked_nodes->find({real_node_with_index, output_info}) != checked_nodes->end()) {
        continue;
      }
      checked_nodes->emplace(real_node_with_index, output_info);
      switch_actor->branch_output_free_index_[output_info.branch_name].emplace_back(i);
      MS_LOG(DEBUG) << "Add free output index:" << i << " for branch:" << output_info.branch_name
                    << " in actor:" << kernel_actor->GetAID();
    }
  }
}

void SuperKernelActor::SetFreePositionForKernelActor() {
  mindspore::HashMap<AnfNodePtr, device::DeviceContextKey> kernel_to_context_key;
  mindspore::HashMap<AnfNodePtr, KernelRunner *> kernel_to_actor;
  std::map<uint32_t, std::vector<CNodePtr>> inplace_groups;

  // 1. Clear free index in actors.
  std::vector<bool> disable_inputs(graph_->input_nodes().size(), false);
  is_input_used_.swap(disable_inputs);

  for (const auto &kernel_actor : kernel_actors_) {
    if (kernel_actor == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(kernel_actor->kernel());
    if (common::AnfAlgo::IsInplaceNode(kernel_actor->kernel(), "inplace_algo")) {
      auto primitive = common::AnfAlgo::GetCNodePrimitive(kernel_actor->kernel());
      MS_EXCEPTION_IF_NULL(primitive);
      auto inplace_group_attr = primitive->GetAttr("inplace_group");
      MS_EXCEPTION_IF_NULL(inplace_group_attr);
      auto group_id = GetValue<uint32_t>(inplace_group_attr);
      inplace_groups[group_id].emplace_back(kernel_actor->kernel());
    }

    kernel_to_actor[kernel_actor->kernel()] = kernel_actor.get();
    if (kernel_actor->device_contexts().empty() || kernel_actor->device_contexts()[0] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid device context, context size:" << kernel_actor->device_contexts().size()
                        << " for actor:" << kernel_actor->GetAID();
    }
    kernel_to_context_key[kernel_actor->kernel()] = kernel_actor->device_contexts()[0]->device_context_key();
    kernel_actor->input_free_index_.clear();
    kernel_actor->output_free_index_.clear();
  }
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    MS_LOG(EXCEPTION) << "Invalid graph device context, context size:" << device_contexts_.size()
                      << " for actor:" << GetAID();
  }
  auto graph_device_context_key = device_contexts_[0]->device_context_key();

  // Get ref count by graph output, the ref count should be increased in launch kernel thread.
  GetRefCountForGraphOutput(output_data_nodes_, output_data_arrows_, kernel_to_actor, inplace_groups, GetAID().Name());

  std::set<std::pair<KernelWithIndex, FreeNodeInfo>> checked_nodes;
  for (auto kernel_iter = kernel_actors_.rbegin(); kernel_iter != kernel_actors_.rend(); ++kernel_iter) {
    const auto &kernel_actor = *kernel_iter;
    if (kernel_actor == nullptr) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(kernel_actor->kernel_);
    if (kernel_actor->device_contexts().empty() || kernel_actor->device_contexts()[0] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid device context, context size:" << kernel_actor->device_contexts().size()
                        << " for actor:" << kernel_actor->GetAID();
    }

    SetInputFreePositionForKernelActor(kernel_actor, kernel_to_context_key, graph_device_context_key, &checked_nodes);
    SetOutputFreePositionForKernelActor(kernel_actor, kernel_to_context_key, graph_device_context_key, &checked_nodes);
  }
  for (const auto &kernel_actor : kernel_actors_) {
    if (kernel_actor == nullptr) {
      continue;
    }
    MS_LOG(DEBUG) << "Actor:" << kernel_actor->GetAID() << " input free index:" << kernel_actor->input_free_index_
                  << " output free index:" << kernel_actor->output_free_index_;
  }
}

void SuperKernelActor::LinkKernelActors() {
  const auto &input_nodes = graph_->input_nodes();
  size_t input_num = input_nodes.size();
  param_node_to_input_idx_.reserve(input_num);
  // Record the parameter first used actor and actor input idx.
  std::vector<std::pair<KernelRunnerPtr, size_t>> param_first_used_kernel_actors(input_num, {nullptr, 0});
  for (size_t i = 0; i < input_num; i++) {
    param_node_to_input_idx_[input_nodes[i].get()] = i;
  }

  input_params_use_cnt_.resize(input_num, 0);

  // 1. Record input index -> device tensor store key (AnfNodePtr), use to check
  // whether a input index of graph input nodes is a persistent device tensor.
  HashMap<size_t, AnfNodePtr> device_tensor_store_keys_map;
  device_tensor_store_keys_map.reserve(device_tensor_store_keys_.size());
  std::for_each(device_tensor_store_keys_.begin(), device_tensor_store_keys_.end(),
                [&device_tensor_store_keys_map](const std::pair<size_t, AnfNodePtr> &item) {
                  device_tensor_store_keys_map.emplace(item.first, item.second);
                });

  HashMap<size_t, ParameterInfo> parameter_indexs_map;
  parameter_indexs_map.reserve(parameter_indexs_.size());
  for (const auto &iter : parameter_indexs_) {
    parameter_indexs_map.emplace(iter.first, iter.second);
  }

  // 2. Record output node -> output index, use to quickly find all output indices of the same output node.
  // Maybe there is same node in all output of graph.
  size_t actor_output_num = output_data_nodes_.size();
  HashMap<AnfNodePtr, std::vector<size_t>> output_node_to_actor_output_index;
  output_node_to_actor_output_index.reserve(actor_output_num);
  for (size_t i = 0; i < actor_output_num; i++) {
    MS_EXCEPTION_IF_NULL(output_data_nodes_[i]);
    if (output_data_nodes_[i]->isa<Parameter>()) {
      output_node_to_actor_output_index[output_data_nodes_[i]].push_back(i);
    }
  }

  // 3. Check input parameter(not persist tensor) as graph output case, need increase the parameter use count for
  // output parameter.
  for (const auto &item : output_node_to_actor_output_index) {
    const auto &output_param = item.first;
    MS_EXCEPTION_IF_NULL(output_param);
    auto input_idx_iter = param_node_to_input_idx_.find(output_param.get());
    if (input_idx_iter == param_node_to_input_idx_.end()) {
      MS_LOG_WITH_NODE(EXCEPTION, output_param)
        << "Can not find index for input node: " << output_param->fullname_with_scope();
    }
    size_t input_node_idx = input_idx_iter->second;
    const auto &output_indices = item.second;
    input_params_use_cnt_.at(input_node_idx) += output_indices.size();
  }

  // 4. Calculate original ref count of CNode and Parameter, prepare input and
  // heterogeneous output device address of all kernels.
  AnalyseNodesDependence(device_tensor_store_keys_map, parameter_indexs_map, output_node_to_actor_output_index,
                         &param_first_used_kernel_actors);

  RecordKernelActorWeight();
  if (IS_OUTPUT_ON(MsLogLevel::kDebug)) {
    for (size_t i = 0; i < input_num; i++) {
      MS_LOG(DEBUG) << "SuperKernelActor: " << GetAID().Name() << " Parameter[" << input_nodes[i]->fullname_with_scope()
                    << "] debug_name: " << input_nodes[i]->DebugString()
                    << " use count is: " << input_params_use_cnt_[i];
    }
  }
}

void ParamFirstUsedKernelActorsToMap(
  const std::vector<std::pair<KernelRunnerPtr, size_t>> &param_first_used_kernel_actors) {
  if (!EnableInputOptimize()) {
    return;
  }
  for (size_t i = 0; i < param_first_used_kernel_actors.size(); ++i) {
    auto &kernel_actor = param_first_used_kernel_actors[i].first;
    auto actor_input_idx = param_first_used_kernel_actors[i].second;
    if (kernel_actor == nullptr) {
      continue;
    }
    kernel_actor->set_is_first_used_param(true, actor_input_idx);
  }
}

namespace {
void UpdateDeviceTensorStoreForKernelRunner(const KernelRunnerPtr &kernel_actor,
                                            const DeviceContext *graph_device_context, const AnfNodePtr &key,
                                            size_t index) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  MS_EXCEPTION_IF_NULL(graph_device_context);
  MS_EXCEPTION_IF_NULL(key);
  if (kernel_actor->device_contexts().empty() || kernel_actor->device_contexts()[0] == nullptr) {
    return;
  }
  const auto &kernel_device_context = kernel_actor->device_contexts()[0];
  if (kernel_device_context->GetDeviceType() == graph_device_context->GetDeviceType()) {
    return;
  }
  if (DeviceTensorStore::GetInstance().Fetch(key.get(), kernel_device_context->GetDeviceType()) != nullptr) {
    return;
  }
  auto kernel_tensors = DeviceTensorStore::GetInstance().Fetch(key.get());
  if (kernel_tensors.empty()) {
    return;
  }
  MS_EXCEPTION_IF_NULL(kernel_tensors[0]);
  auto kernel_tensor = SchedulerHelper::CloneKernelTensorWithDeviceInfo(kernel_tensors[0], kernel_device_context);
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  SchedulerHelper::AddDeviceTensorStore(key, kernel_tensor);
  MS_LOG(DEBUG) << "Add device tensor copy store for kernel actor:" << kernel_actor->GetAID()
                << " input index:" << index << " key:" << key->DebugString() << " node addr:" << key
                << " kernel tensor:" << kernel_tensor->ToString();
}
}  // namespace

void SuperKernelActor::AnalyseNodesDependence(
  const HashMap<size_t, AnfNodePtr> &device_tensor_store_keys_map,
  const HashMap<size_t, ParameterInfo> &parameter_indexs_map,
  const HashMap<AnfNodePtr, std::vector<size_t>> &output_node_to_actor_output_index,
  std::vector<std::pair<KernelRunnerPtr, size_t>> *param_first_used_kernel_actors) {
  const auto &execution_order = graph_->execution_order();
  mindspore::HashMap<size_t, mindspore::HashMap<size_t, KernelRunnerPtr>> param_first_used_actors_on_stream;
  std::map<std::pair<size_t, size_t>, size_t> parameter_used_times;
  size_t kernel_num = execution_order.size();
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel = execution_order[i];
    MS_EXCEPTION_IF_NULL(kernel);
    if (!IsKernelActor(kernel, GraphExecutionStrategy::kPipeline) || IsSkippedKernelActor(kernel)) {
      continue;
    }

    auto kernel_input_num = common::AnfAlgo::GetInputTensorNum(kernel);
    for (size_t j = 0; j < kernel_input_num; j++) {
      auto input_node_with_idx = common::AnfAlgo::GetPrevNodeOutput(kernel, j, false);
      MS_EXCEPTION_IF_NULL(input_node_with_idx.first);

      if (input_node_with_idx.first->isa<CNode>()) {
        if (IsSkippedKernelActor(input_node_with_idx.first)) {
          auto real_input_node_with_idx = common::AnfAlgo::GetPrevNodeOutput(input_node_with_idx.first, 0, false);
          if (!real_input_node_with_idx.first->isa<CNode>()) {
            MS_INTERNAL_EXCEPTION(RuntimeError)
              << "Expect a CNode for input[0] of kernel: " << input_node_with_idx.first->fullname_with_scope()
              << ", which is a skipped kernel, but got: " << real_input_node_with_idx.first->DebugString();
          }
          input_node_with_idx = real_input_node_with_idx;
        }
        LinkKernelActor(kernel, j, input_node_with_idx.first, input_node_with_idx.second);
      } else if (input_node_with_idx.first->isa<ValueNode>()) {
        auto device_tensor_store_key = AnfAlgo::FetchFrontNodeByBackendNode(input_node_with_idx.first, *graph_);
        MS_EXCEPTION_IF_NULL(device_tensor_store_key);
        auto &kernel_actor = kernel_actors_[i];
        MS_EXCEPTION_IF_NULL(kernel_actor);
        (void)kernel_actor->device_tensor_store_keys_.emplace_back(j, device_tensor_store_key);
      } else if (input_node_with_idx.first->isa<Parameter>()) {
        auto input_idx_iter = param_node_to_input_idx_.find(input_node_with_idx.first.get());
        if (input_idx_iter == param_node_to_input_idx_.end()) {
          MS_LOG_WITH_NODE(EXCEPTION, input_node_with_idx.first)
            << "Can not find index for input node: " << input_node_with_idx.first->fullname_with_scope();
        }
        size_t input_node_idx = input_idx_iter->second;
        if (!IsOnlyDependShape(kernel, j)) {
          ++(input_params_use_cnt_.at(input_node_idx));
        }

        const auto &device_tensor_store_key_iter = device_tensor_store_keys_map.find(input_node_idx);
        if (device_tensor_store_key_iter != device_tensor_store_keys_map.end()) {
          auto &kernel_actor = kernel_actors_[i];
          MS_EXCEPTION_IF_NULL(kernel_actor);
          UpdateDeviceTensorStoreForKernelRunner(kernel_actor, device_contexts_[0],
                                                 device_tensor_store_key_iter->second, j);
          (void)kernel_actor->device_tensor_store_keys_.emplace_back(j, device_tensor_store_key_iter->second);
        }

        const auto &parameter_index_iter = parameter_indexs_map.find(input_node_idx);
        if (parameter_index_iter != parameter_indexs_map.end()) {
          auto &kernel_actor = kernel_actors_[i];
          MS_EXCEPTION_IF_NULL(kernel_actor);
          (void)kernel_actor->parameter_indexs_.emplace_back(j, parameter_index_iter->second);
          auto outer_index = parameter_index_iter->second.second;
          auto inner_index = parameter_index_iter->second.first.second;
          parameter_used_times[{outer_index, inner_index}]++;
          SetParamFirstUsedKernelActors(input_node_idx, j, &kernel_actors_[i], param_first_used_kernel_actors,
                                        &param_first_used_actors_on_stream);
        }

        if (enable_input_optimize_) {
          if (device_tensor_store_key_iter == device_tensor_store_keys_map.end() &&
              parameter_index_iter == parameter_indexs_map.end()) {
            kernel_input_to_graph_input_indices_[kernel.get()].emplace_back(j, input_node_idx);
          }
        } else {
          if (device_tensor_store_key_iter == device_tensor_store_keys_map.end()) {
            kernel_input_to_graph_input_indices_[kernel.get()].emplace_back(j, input_node_idx);
          }
        }

        auto output_idx_iter = output_node_to_actor_output_index.find(input_node_with_idx.first);
        if (output_idx_iter != output_node_to_actor_output_index.end()) {
          kernel_input_to_actor_output_indices_[kernel.get()].emplace_back(j, output_idx_iter->second);
          MS_VLOG(VL_RUNTIME_FRAMEWORK_KERNEL)
            << "Add kernel input:" << j << " kernel:" << kernel->fullname_with_scope() << " for actor:" << GetAID();
        }
      }
    }
  }

  CollectStreamFirstUsedParamKernelActors(&param_first_used_actors_on_stream);
  ParamFirstUsedKernelActorsToMap(*param_first_used_kernel_actors);
  RecordInputParamsWithoutUser(graph_, parameter_indexs_map, input_params_use_cnt_, &input_params_no_user_);
  CalculateParameterUsedTimes(parameter_used_times);
}

void SuperKernelActor::RecordKernelActorWeight() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  enable_infer_boost_ = ms_context->IsEnableInferBoost();
  in_increment_ = enable_infer_boost_ && (graph_phase_.find("increment") != std::string::npos);
  if (!EnableInputOptimize() || !enable_infer_boost_) {
    return;
  }
  const auto &execution_order = graph_->execution_order();
  size_t kernel_num = execution_order.size();
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel = execution_order[i];
    MS_EXCEPTION_IF_NULL(kernel);

    auto kernel_input_num = common::AnfAlgo::GetInputTensorNum(kernel);
    auto &kernel_actor = kernel_actors_[i];
    MS_EXCEPTION_IF_NULL(kernel_actor);
    kernel_actor->is_weight_.resize(kernel_input_num, false);
    for (const auto &iter : kernel_actor->parameter_indexs_) {
      auto input_index = iter.first;
      auto node = iter.second.first.first;
      MS_EXCEPTION_IF_NULL(node);
      if (node->isa<Parameter>() && common::AnfAlgo::IsParameterWeight(node->cast<ParameterPtr>())) {
        kernel_actor->is_weight_[input_index] = true;
      }
    }
  }
}

void SuperKernelActor::LinkKernelActor(const CNodePtr &kernel, size_t input_index, const AnfNodePtr &input_kernel,
                                       size_t output_index) {
  // Shape depend kernel should not increase ref count.
  if (IsOnlyDependShape(kernel, input_index)) {
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_kernel, output_index, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    kernel_tensor->UpdateFlag(device::kDeviceAddressFlagNullptr);

    auto *kernel_actor = cnode_to_kernel_actor_[kernel];
    MS_EXCEPTION_IF_NULL(kernel_actor);
    kernel_actor->SetInputDeviceTensor(kernel_tensor, input_index);
    kernel_actor->memory_free_list_[input_index] = kernel_tensor;
    return;
  }

  LinkKernelActorByDeviceType(kernel, input_index, input_kernel, output_index);
}

void SuperKernelActor::LinkKernelActorByDeviceType(const CNodePtr &kernel, size_t input_index,
                                                   const AnfNodePtr &input_kernel, size_t output_index) {
  MS_EXCEPTION_IF_NULL(kernel);
  MS_EXCEPTION_IF_NULL(input_kernel);
  auto *kernel_actor = cnode_to_kernel_actor_[kernel];
  MS_EXCEPTION_IF_NULL(kernel_actor);
  const auto *device_context = kernel_actor->device_contexts().front();
  MS_EXCEPTION_IF_NULL(device_context);

  auto *input_kernel_actor = cnode_to_kernel_actor_[input_kernel];
  if (input_kernel_actor == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Not found kernel actor for node: " + input_kernel->fullname_with_scope();
  }
  const auto *input_device_context = input_kernel_actor->device_contexts().front();
  MS_EXCEPTION_IF_NULL(input_device_context);

  const auto &input_kernel_tensor = AnfAlgo::GetOutputKernelTensor(input_kernel, output_index, false);
  MS_EXCEPTION_IF_NULL(input_kernel_tensor);
  const auto &input_device_tensor = input_kernel_tensor->device_address();
  MS_EXCEPTION_IF_NULL(input_device_tensor);

  bool need_not_copy_output_device_addr = (device_context->GetDeviceType() == input_device_context->GetDeviceType()) ||
                                          SchedulerHelper::IsIgnoredInputAddressV2(kernel_actor, input_index);
  MS_LOG(DEBUG) << "Kernel:" << kernel->fullname_with_scope() << " input kernel:" << input_kernel->fullname_with_scope()
                << " input index:" << input_index << " device context type:" << device_context->GetDeviceType()
                << " input context type:" << input_device_context->GetDeviceType()
                << " need copy:" << need_not_copy_output_device_addr << " for actor:" << GetAID();
  if (need_not_copy_output_device_addr) {
    if (input_index >= kernel_actor->input_kernel_tensors_.size() ||
        input_index >= kernel_actor->input_kernel_tensors_for_infer_.size() ||
        input_index >= kernel_actor->memory_free_list_.size()) {
      MS_LOG(EXCEPTION) << "Invalid input index:" << input_index
                        << " for input size:" << kernel_actor->input_kernel_tensors_.size()
                        << "  kernel tensor size:" << kernel_actor->input_kernel_tensors_for_infer_.size()
                        << " memory free list size:" << kernel_actor->memory_free_list_.size()
                        << " for actor:" << kernel_actor->GetAID();
    }
    kernel_actor->SetInputDeviceTensor(input_kernel_tensor, input_index);
    kernel_actor->memory_free_list_[input_index] = input_kernel_tensor;
    return;
  }

  if (enable_trace_memory_ && graph_phase_.find("increment") != std::string::npos) {
    MS_LOG(EXCEPTION)
      << "Find heterogeneous operators in graph: " << graph_->ToString() << ", the input[" << input_index << "] of ["
      << device_context->device_context_key().device_type_ << "] op(" << kernel->fullname_with_scope() << ") is ["
      << input_device_context->device_context_key().device_type_ << "] op(" << input_kernel->fullname_with_scope()
      << "). Trace memory feature can not work in this case. Please eliminate heterogeneity(cpu "
         "kernel), or disable trace memory feature by export MS_ENABLE_TRACE_MEMORY=off. Note: Disabling the trace "
         "memory feature will degrade memory management performance. Additionally, it will automatically disable "
         "the kernel group launch(parallel launch) and graph capture features, which may reduce network execution "
         "performance.";
  }

  auto &copy_output_kernel_tensors = input_kernel_actor->copy_output_kernel_tensors_;
  auto iter = copy_output_kernel_tensors.find(output_index);
  if (iter == copy_output_kernel_tensors.end()) {
    const auto input_copy_kernel_tensor =
      SchedulerHelper::CloneKernelTensorWithDeviceInfo(input_kernel_tensor, device_context);
    MS_EXCEPTION_IF_NULL(input_copy_kernel_tensor);
    MS_LOG(DEBUG) << "Create kernel tensor:" << input_copy_kernel_tensor->ToString();
    input_copy_kernel_tensor->set_device_ptr(nullptr);

    auto input_copy_device_address = input_copy_kernel_tensor->device_address();
    MS_LOG(DEBUG) << "Create copy device address:" << input_copy_device_address
                  << " for actor:" << input_kernel_actor->GetAID();
    auto ret_pair = copy_output_kernel_tensors.emplace(
      output_index,
      std::make_pair(input_copy_kernel_tensor, std::make_pair(device_context, std::vector<KernelTensorPtr>())));
    if (ret_pair.second) {
      iter = ret_pair.first;
    } else {
      MS_LOG(EXCEPTION) << "Insert copy output device address failed.";
    }
  }

  const auto &input_copy_kernel_tensor = iter->second.first;
  MS_EXCEPTION_IF_NULL(input_copy_kernel_tensor);
  auto input_copy_device_address = input_copy_kernel_tensor->device_address().get();
  MS_EXCEPTION_IF_NULL(input_copy_device_address);
  if (kernel_actor->modifiable_ref_input_indexes_.count(input_index) > 0) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Add device tensor copy store for device address:" << input_copy_device_address
      << " type:" << input_copy_device_address->GetDeviceType() << " and " << input_device_tensor
      << " type:" << input_device_tensor->GetDeviceType() << " for copy actor:" << GetAID();
    if (kernel_actor->kernel_info_ != nullptr) {
      const auto &ref_map = kernel_actor->kernel_info_->out_in_ref_map();
      auto index_iter =
        std::find_if(ref_map.begin(), ref_map.end(),
                     [input_index](const std::pair<size_t, size_t> &pair) { return pair.second == input_index; });
      if (index_iter != ref_map.end() && kernel_actor->output_kernel_tensors_.size() > index_iter->first &&
          kernel_actor->output_kernel_tensors_[index_iter->first] != nullptr) {
        iter->second.second.second.emplace_back(kernel_actor->output_kernel_tensors_[index_iter->first]);
        MS_LOG(DEBUG) << "Add dst device address:"
                      << kernel_actor->output_kernel_tensors_[index_iter->first]->device_address().get()
                      << " for input copy device address:" << input_copy_device_address
                      << " for actor:" << kernel_actor->GetAID();
      }
    }
  }
  kernel_actor->SetInputDeviceTensor(input_copy_kernel_tensor, input_index);
  kernel_actor->memory_free_list_[input_index] = input_copy_kernel_tensor;
}

void SuperKernelActor::TrackInputMemory() {
  if (!device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    return;
  }

  for (auto &kernel_tensor : input_kernel_tensors_) {
    if (kernel_tensor == nullptr || kernel_tensor->device_address() == nullptr ||
        !kernel_tensor->device_address()->IsPtrValid()) {
      continue;
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(UseMemBlock, GetAID().Name(),
                                                   kernel_tensor->device_address()->GetPtr());
  }
}

void SuperKernelActor::IncreaseNewRefCounts(OpContext<KernelTensor> *const context) {
  if (enable_kbk_sub_graph_execute_) {
    MS_LOG(DEBUG) << "Skip increaase new ref count for actor:" << GetAID();
    return;
  }
  std::for_each(output_data_.begin(), output_data_.end(),
                [this](const auto &pair) { IncreaseNewRefCount(pair.first.get()); });
}
}  // namespace runtime
}  // namespace mindspore
