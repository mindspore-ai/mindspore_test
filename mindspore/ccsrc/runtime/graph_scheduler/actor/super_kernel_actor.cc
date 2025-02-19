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

#include <set>
#include <algorithm>
#include "include/backend/mem_reuse/mem_tracker.h"
#include "runtime/graph_scheduler/actor/super_kernel_actor.h"
#include "runtime/graph_scheduler/scheduler_helper.h"
#include "runtime/graph_scheduler/actor/output_actor.h"
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "runtime/graph_scheduler/actor/debug_actor.h"
#include "runtime/device/multi_stream_controller.h"
#include "runtime/pipeline/task/batch_launch_kernel_task.h"
#include "runtime/runtime_conf/runtime_conf.h"
#include "async/async.h"
#include "utils/phase.h"
#include "utils/llm_manager.h"
#include "utils/log_adapter.h"
#include "op_def/framework_ops.h"
#include "pybind_api/gil_scoped_long_running.h"

namespace mindspore {
namespace runtime {
size_t SuperKernelActor::parallel_dispatch_num_ = 2;
size_t SuperKernelActor::parallel_slice_num_ = 4;

std::vector<std::pair<size_t, void *>> SuperKernelActor::streams_;
std::vector<DeviceEventPtr> SuperKernelActor::events_;
std::vector<AsyncRQueuePtr> SuperKernelActor::queues_;

namespace {
inline void UpdateShape(const AnfNodePtr &input_node, const DeviceTensorPtr &node_device_tensor,
                        DeviceTensor *input_device_tensor, const KernelTransformType &type) {
  MS_EXCEPTION_IF_NULL(input_node);
  const auto &node_device_kernel_tensor = node_device_tensor->kernel_tensor();
  MS_EXCEPTION_IF_NULL(input_device_tensor);
  const auto &input_kernel_tensor = input_device_tensor->kernel_tensor();
  MS_EXCEPTION_IF_NULL(node_device_kernel_tensor);
  MS_EXCEPTION_IF_NULL(input_kernel_tensor);
  if (type != KernelTransformType::kSuperKernelActor || input_node->cast<ParameterPtr>()->has_dynamic_shape()) {
    // For dynamic shape in sub graph sink and any type parameter, the input size should be updated.
    node_device_tensor->SetSize(input_device_tensor->GetSize());
    // Update Shape.
    node_device_kernel_tensor->SetShape(input_kernel_tensor->GetShape()->Clone());
  }
}

inline bool InputDataNoNeedCopy(const AnfNodePtr &input_node, DeviceTensor *input_device_tensor,
                                const DeviceTensorPtr &node_device_tensor, const KernelTransformType &type) {
  if (input_device_tensor == nullptr) {
    return true;
  }

  if (input_device_tensor == node_device_tensor.get()) {
    (void)input_device_tensor->TouchSyncHandler();
    return true;
  }

  UpdateShape(input_node, node_device_tensor, input_device_tensor, type);

  if (TEST_FLAG(node_device_tensor->flag(), device::kDeviceAddressFlagNotUsed) ||
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
  size_t graph_input_index, size_t actor_input_index, KernelActorPtr *kernel_actor,
  std::vector<std::pair<KernelActorPtr, size_t>> *param_first_used_kernel_actors,
  mindspore::HashMap<size_t, mindspore::HashMap<size_t, KernelActorPtr>> *param_first_used_actors_on_stream) {
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
    (*param_first_used_kernel_actors)[graph_input_index].first = *kernel_actor;
    (*param_first_used_kernel_actors)[graph_input_index].second = actor_input_index;
  }
}

void CollectStreamFirstUsedParamKernelActors(
  mindspore::HashMap<size_t, mindspore::HashMap<size_t, KernelActorPtr>> *param_first_used_actors_on_stream,
  mindspore::HashSet<KernelActor *> *kernel_actors_insert_event) {
  if (!EnableInputOptimize()) {
    return;
  }
  for (const auto &iter : *param_first_used_actors_on_stream) {
    const auto &stream_with_kernel_actors = iter.second;
    for (const auto &stream_with_actor_iter : stream_with_kernel_actors) {
      (*kernel_actors_insert_event).insert(stream_with_actor_iter.second.get());
    }
  }
}

void ParamFirstUsedKernelActorsToMap(
  const std::vector<std::pair<KernelActorPtr, size_t>> &param_first_used_kernel_actors,
  mindspore::HashMap<KernelActorPtr, std::vector<std::pair<size_t, size_t>>> *kernel_actor_to_graph_parameters_map) {
  if (!EnableInputOptimize()) {
    return;
  }
  for (size_t i = 0; i < param_first_used_kernel_actors.size(); ++i) {
    auto &kernel_actor = param_first_used_kernel_actors[i].first;
    auto actor_input_idx = param_first_used_kernel_actors[i].second;
    if (kernel_actor == nullptr) {
      continue;
    }
    const auto &iter = (*kernel_actor_to_graph_parameters_map).find(kernel_actor);
    if (iter == (*kernel_actor_to_graph_parameters_map).end()) {
      (*kernel_actor_to_graph_parameters_map)[kernel_actor].emplace_back(actor_input_idx, i);
    } else {
      auto &param_map_list = iter->second;
      param_map_list.push_back({actor_input_idx, i});
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
    if (input_params_use_cnt.at(i) == 0) {
      const auto &parameter_index_iter = parameter_indexs_map.find(i);
      if (parameter_index_iter != parameter_indexs_map.end()) {
        input_params_no_user->emplace(i, parameter_index_iter->second);
      }
    }
  }
}
}  // namespace

SuperKernelActor::~SuperKernelActor() {
  if (!queues_.empty()) {
    for (auto &q : queues_) {
      q->WorkerJoin();
    }
    queues_.clear();
  }
  if (!events_.empty()) {
    events_.clear();
  }
  if (!parallel_launch_kernels_.empty()) {
    parallel_launch_kernels_.clear();
  }
  if (!serial_launch_kernels_.empty()) {
    serial_launch_kernels_.clear();
  }
}

void SuperKernelActor::Init() {
  MS_EXCEPTION_IF_NULL(graph_);
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
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
    auto device_address = AnfAlgo::GetMutableOutputAddr(output_node, IntToSize(data_arrow->from_output_index_), false);
    data->data_ = device_address.get();
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
      auto device_address = AnfAlgo::GetMutableOutputAddr(output_node, output_with_index.second, false);
      MS_EXCEPTION_IF_NULL(device_address);
      if (device_address->is_ptr_persisted() || graph_->is_dynamic_shape()) {
        MS_LOG(DEBUG) << "Actor:" << GetAID() << " skip alloc memory for device address:" << device_address
                      << " is persist:" << device_address->is_ptr_persisted()
                      << " is dynamic shape:" << graph_->is_dynamic_shape()
                      << " output node:" << output_node->DebugString();
        continue;
      }
      // Free the ptr in device address of output node.
      if (device_address->GetPtr() != nullptr) {
        MS_LOG(INFO) << "Output node:" << output_node->DebugString() << " has a default ptr, maybe a mem leak.";
        device_address->set_ptr(nullptr);
      }
      if (common::IsDryRun()) {
        device_address_to_node_[device_address.get()] = {device_address->GetSize(), output_node->fullname_with_scope()};
      }
      memory_alloc_list_.emplace_back(device_address.get());
    }
  }

  // Check whether the parameter needs to be copied out.
  node_device_tensors_.resize(graph_->input_nodes().size());
  is_parameters_need_copy_.resize(graph_->input_nodes().size());
  copy_input_device_tensors_.resize(graph_->input_nodes().size());
  for (size_t i = 0; i < graph_->input_nodes().size(); ++i) {
    const auto &input_node = graph_->input_nodes()[i];
    MS_EXCEPTION_IF_NULL(input_node);
    node_device_tensors_[i] = AnfAlgo::GetMutableOutputAddr(input_node, 0, false);
    if (!common::AnfAlgo::HasAbstractRef(input_node)) {
      is_parameters_need_copy_[i] = false;
      continue;
    }
    // If the parameter has ref attribute and is directly used by the kernel in the graph, it needs to be copied.
    is_parameters_need_copy_[i] = true;
  }

  if (type_ == KernelTransformType::kSuperKernelActor && !enable_kbk_sub_graph_execute_) {
    MS_EXCEPTION_IF_NULL(device_contexts_[0]);
    MS_EXCEPTION_IF_NULL(device_contexts_[0]->graph_executor_);
    device_contexts_[0]->graph_executor_->InitGraphInfo(graph_);
  }
}

void SuperKernelActor::InitParallelDispatchResource() {
  if (streams_.empty()) {
    streams_.resize(parallel_dispatch_num_);
    for (size_t i = 0; i < parallel_dispatch_num_; i++) {
      if (!device_contexts_[0]->device_res_manager_->CreateStream(&(streams_[i].first))) {
        MS_LOG(EXCEPTION) << "Create stream failed.";
      }
      streams_[i].second = device_contexts_[0]->device_res_manager_->GetStream(streams_[i].first);
      MS_EXCEPTION_IF_NULL(streams_[i].second);
    }
  }

  if (events_.empty()) {
    // New one more for sync between default stream and last launch stream;
    for (size_t i = 0; i < parallel_dispatch_num_ * parallel_slice_num_ + 1; i++) {
      auto event = device_contexts_[0]->device_res_manager_->CreateEventWithFlag(false, false, false);
      MS_EXCEPTION_IF_NULL(event);
      events_.push_back(event);
    }
  }

  if (queues_.empty()) {
    for (size_t i = 0; i < parallel_dispatch_num_; i++) {
      auto queue = std::make_unique<AsyncRQueue>(std::string("batch_launch_") + std::to_string(i),
                                                 runtime::kThreadWaitLevel::kLevelDevice);
      MS_EXCEPTION_IF_NULL(queue);
      queue->SetSpin(true);
      queues_.push_back(std::move(queue));
    }
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

void SuperKernelActor::CorrectRefCountByCondition(size_t index, DeviceTensor *device_tensor,
                                                  std::vector<DeviceTensor *> *memory_free_list) {
  // There is no memory free action for use trace memory step, need to free input device address of the kernel graph
  // after launch all kernels.
  if (ActorDispatcher::enable_use_trace_memory()) {
    if ((device_tensor->original_ref_count() != SIZE_MAX || device_tensor->dynamic_ref_count() != INT32_MAX)) {
      (void)(*memory_free_list).emplace_back(device_tensor);
    }
  } else {
    CorrectRefCount(index, device_tensor);
  }
}

void SuperKernelActor::FetchInputDeviceTensor(OpContext<DeviceTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                  "Invalid device context for super kernel actor:" + GetAID().Name());
  }
  std::vector<DeviceTensor *> memory_free_list;
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      MS_EXCEPTION_IF_NULL(input_data->data_);
      size_t index = IntToSize(input_data->index_);
      if (index >= input_device_tensors_.size()) {
        std::string error_info = "Invalid input index:" + std::to_string(index) +
                                 " total:" + std::to_string(input_device_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      input_device_tensors_[index] = input_data->data_;

      if (IsNeedProfilieMemoryLog()) {
        auto output_address = reinterpret_cast<std::uintptr_t>(input_device_tensors_[index]);
        MS_LOG(WARNING) << "Need Profile Memory, Memory use, actor name: " << GetAID().Name()
                        << ", kernel graph: " << graph_->ToString() << ", device address class ptr: " << output_address
                        << ", device address size: " << input_device_tensors_[index]->GetSize()
                        << ", device address addr: " << input_device_tensors_[index]->GetPtr() << ", index: " << index;
      }

      if (!enable_kbk_sub_graph_execute_) {
        if (input_data->data_->dynamic_ref_count() != INT32_MAX) {
          (void)memory_free_list.emplace_back(input_data->data_);
        }
        continue;
      }

      CorrectRefCountByCondition(index, input_data->data_, &memory_free_list);
    }
    memory_free_lists_.push(memory_free_list);
  }
}

void SuperKernelActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);
  if (device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), "SuperKernelActor", graph_->ToString());
  }

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
  }
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Invalid device context for super kernel actor:" + GetAID().Name());
  }
  MS_LOG(INFO) << "Super kernel actor(" << GetAID().Name()
               << ") launches graph: " << std::to_string(graph_->graph_id());
  if (IsNeedProfilieMemoryLog()) {
    MS_LOG(WARNING) << "Need Profile Memory, launch actor name: " << GetAID().Name()
                    << ", kernel graph: " << graph_->ToString();
  }
  if (!WaitRuntimePipelineFinish(context)) {
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
    for (auto &device_tensor : memory_alloc_list_) {
      MS_EXCEPTION_IF_NULL(device_tensor);
      if (device_tensor->IsNotNeedAlloc()) {
        continue;
      }
      if (IsNeedProfilieMemoryLog()) {
        auto &info = device_address_to_node_[device_tensor];
        auto output_address = reinterpret_cast<std::uintptr_t>(device_tensor);
        MS_LOG(WARNING) << "Need Profile Memory, Memory need allocated, actor name: " << GetAID().Name()
                        << ", kernel graph: " << graph_->ToString() << ", node: " << info.node_full_name
                        << ", device address class ptr: " << output_address << ", device address size: " << info.size;
      }
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
        AddMemInfo, GetAID().Name(), device::tracker::MemType::kGraphOutput, device_tensor->GetSize(), device_tensor);
    }
    SendMemoryAllocReq(context);
  } else {
    OnMemoryAllocFinish(context);
  }
  if (IsNeedProfilieMemoryLog()) {
    MS_LOG(WARNING) << "Need Profile Memory, end launch, actor name: " << GetAID().Name()
                    << ", kernel graph: " << graph_->ToString();
  }
}

void SuperKernelActor::FetchPersistentDeviceTensor() {
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    auto input_device_tensor = DeviceTensorStore::GetInstance()
                                 .Fetch(device_tensor_store_key.second.get(), device_contexts_[0]->GetDeviceType())
                                 .get();
    // Ge backend maybe nullptr.
    if (input_device_tensor == nullptr) {
      MS_LOG(DEBUG) << "Failed get device tensor for node:" << device_tensor_store_key.second->DebugString()
                    << " index:" << device_tensor_store_key.first;
      continue;
    }

    size_t index = device_tensor_store_key.first;
    input_device_tensors_[index] = input_device_tensor;
  }
}

void SuperKernelActor::CorrectRefCount(size_t input_index, DeviceTensor *device_tensor) {
  MS_EXCEPTION_IF_NULL(device_tensor);
  if (device_tensor->original_ref_count() == SIZE_MAX && device_tensor->dynamic_ref_count() == INT32_MAX) {
    return;
  }

  const auto &input_use_cnt = input_params_use_cnt_.at(input_index);
  if (input_use_cnt == 0) {
    if (device_tensor->original_ref_count() != SIZE_MAX) {
      // No user for this input in graph.
      MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(device_tensor, device_contexts_[0], GetAID().Name());
    }
    return;
  }

  if (device_tensor->original_ref_count() != SIZE_MAX) {
    device_tensor->IncreaseRefCount(input_use_cnt);
  } else if (device_tensor->dynamic_ref_count() != INT32_MAX) {
    device_tensor->IncreaseDynamicRefCount(GetAID().Name(), SizeToInt(input_use_cnt));
  }
  // Need to decrease current ref count once.
  MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(device_tensor, device_contexts_[0], GetAID().Name());
}

void SuperKernelActor::UpdateMemoryTraceMangerStatus(OpContext<DeviceTensor> *const context) {
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
          MS_LOG(DEBUG) << "Not found kernel block info for kernel: " << kernel->fullname_with_scope()
                        << ", is output kernel: " << kernel_actor->is_output_kernel_;
        } else {
          const auto &kernel_mem_block = iter->second;
          for (auto &block : kernel_mem_block) {
            MS_EXCEPTION_IF_NULL(block);
            if (block->mem_type_ == kOutputMem) {
              kernel_actor->output_kernel_tensors_.at(block->index_)->set_device_ptr(nullptr);
            } else {
              kernel_actor->workspace_kernel_tensors_.at(block->index_)->set_device_ptr(nullptr);
            }
          }
        }
      }
    }

    // First step for dynamic shape, need to record memory trace.
    MemoryTraceManager::GetInstance().Clear();
    static const size_t memory_block_size = 3000;
    MemoryTraceManager::GetInstance().ReserveKernelMemoryBlocks(memory_block_size, device_contexts_[0]);
  } else {
    // Not first step for dynamic shape, use record trace memory.
    AllocateTraceMemory(context);
  }
}

void SuperKernelActor::SetTraceMemoryForKernel(const KernelActorPtr &kernel_actor, bool safe_update) {
  const auto &kernel = kernel_actor->kernel();
  MS_EXCEPTION_IF_NULL(kernel);

  // Allocate trace memory for static memory step.
  const std::shared_ptr<mindspore::HashMap<CNodePtr, std::vector<KernelMemoryTraceBlockPtr>>> &all_kernel_block_info =
    MemoryTraceManager::GetInstance().GetAllKernelBlocksnfo();
  MS_EXCEPTION_IF_NULL(all_kernel_block_info);
  const auto &iter = all_kernel_block_info->find(kernel);
  if (iter == all_kernel_block_info->end()) {
    MS_LOG(DEBUG) << "Not found kernel block info for kernel: " << kernel->fullname_with_scope()
                  << ", is output kernel: " << kernel_actor->is_output_kernel_;
  } else {
    const auto &kernel_mem_block = iter->second;
    const auto &merge_blocks_with_device_context = MemoryTraceManager::GetInstance().GetMergeBlocks();
    MS_EXCEPTION_IF_NULL(merge_blocks_with_device_context);
    const auto &merge_blocks = merge_blocks_with_device_context->at(kernel_actor->device_contexts_[0]);
    for (auto &block : kernel_mem_block) {
      MS_EXCEPTION_IF_NULL(block);
      void *ptr = merge_blocks.at(block->in_memory_trace_block_index_)->start_ + block->offset_in_memory_trace_block_;
      MS_EXCEPTION_IF_NULL(ptr);
      if (block->mem_type_ == kOutputMem) {
        if (!safe_update) {
          kernel_actor->output_kernel_tensors_.at(block->index_)->set_device_ptr(ptr);
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

void SuperKernelActor::SetInputTraceMemory(const KernelActorPtr &kernel_actor) const {
  const auto &merge_blocks_with_device_context = MemoryTraceManager::GetInstance().GetMergeBlocks();
  MS_EXCEPTION_IF_NULL(merge_blocks_with_device_context);
  const auto &merge_blocks = merge_blocks_with_device_context->at(kernel_actor->device_contexts_[0]);

  const auto &kernel_tensor_to_kernel_mem_blocks = MemoryTraceManager::GetInstance().GetKernelTensorToMemBlocksInfo();
  MS_EXCEPTION_IF_NULL(kernel_tensor_to_kernel_mem_blocks);

  for (auto &input_kernel_tensor : kernel_actor->input_kernel_tensors_) {
    const auto &iter = kernel_tensor_to_kernel_mem_blocks->find(input_kernel_tensor);
    if (iter == kernel_tensor_to_kernel_mem_blocks->end()) {
      continue;
    }
    auto &kernel_mem_block = iter->second;
    void *ptr = merge_blocks.at(kernel_mem_block->in_memory_trace_block_index_)->start_ +
                kernel_mem_block->offset_in_memory_trace_block_;

    std::lock_guard<SpinLock> lock(kernel_mem_block->lock_);
    if (input_kernel_tensor->device_ptr() != ptr) {
      input_kernel_tensor->set_device_ptr(ptr);
    }
  }
}

void SuperKernelActor::AllocateTraceMemory(OpContext<DeviceTensor> *const context) const {
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
        SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *context, *(device_contexts_[0]),
                                                    GetAID().Name(), block->size_);
      }
      block->start_ = reinterpret_cast<uint8_t *>(block_addr);
    }
  }
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
}

bool SuperKernelActor::CopyHeterogeneousOutput(OpContext<DeviceTensor> *const context,
                                               const KernelActorPtr &kernel_actor) const {
  if (!WaitRuntimePipelineFinish(context)) {
    return false;
  }

  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kCopyData, GetAID().Name());
  for (const auto &output_index_to_copy_address : kernel_actor->copy_output_device_tensors_) {
    const auto &output_index = output_index_to_copy_address.first;
    const auto &dest_device_address = output_index_to_copy_address.second.first.get();
    const auto &dest_device_context = output_index_to_copy_address.second.second.first;
    const auto &src_device_address = kernel_actor->output_device_tensors_.at(output_index);
    const auto &src_device_context = kernel_actor->device_contexts_[0];
    const auto &ref_output_device_address = output_index_to_copy_address.second.second.second;

    if (kernel_actor->is_dynamic_shape_) {
      // For dynamic shape case.
      const auto &dest_kernel_tensor = dest_device_address->kernel_tensor();
      const auto &src_kernel_tensor = src_device_address->kernel_tensor();
      MS_EXCEPTION_IF_NULL(dest_kernel_tensor);
      MS_EXCEPTION_IF_NULL(src_kernel_tensor);
      dest_kernel_tensor->SetType(src_kernel_tensor->GetType()->Clone());
      dest_kernel_tensor->SetShape(src_kernel_tensor->GetShape()->Clone());
      dest_kernel_tensor->set_size(src_kernel_tensor->size());
    }

    // Allocate memory.
    if (dest_device_address->kernel_tensor()->device_ptr() != nullptr) {
      if (ref_output_device_address.empty()) {
        MS_LOG_WITH_NODE(EXCEPTION, kernel_actor->kernel_)
          << "Memory leak detected in copy output device address for kernel: "
          << kernel_actor->kernel_->fullname_with_scope();
      }
      dest_device_context->device_res_manager_->FreeMemory(dest_device_address);
    }
    std::vector<DeviceTensor *> mem_alloc_list = {dest_device_address};
    MemoryManagerActor::GetInstance()->AllocateMemory(&mem_alloc_list, dest_device_context, context,
                                                      kernel_actor->GetAID());
    if (IsRunningFailed(context)) {
      // Maybe allocate memory failed, early stop to run graph.
      return false;
    }

    auto ret = Copy(dest_device_address, src_device_address);
    if (!ret) {
      MS_LOG(ERROR) << "Copy for heterogeneous output failed, kernel actor: " << kernel_actor->GetAID().Name()
                    << ", output index: " << output_index << ", dest device address: " << dest_device_address
                    << ", src device address: " << src_device_address;
      return false;
    }
    if (!ref_output_device_address.empty()) {
      MS_LOG(DEBUG) << "Add device tensor copy store for device address:" << src_device_address
                    << " type:" << src_device_address->GetDeviceType() << " and " << dest_device_address
                    << " type:" << dest_device_address->GetDeviceType() << " for actor:" << GetAID();
      DeviceTensorCopyStore::GetInstance().Insert(src_device_address, dest_device_address);
      for (const auto &ref_device_address : ref_output_device_address) {
        MS_EXCEPTION_IF_NULL(ref_device_address);
        if (ref_device_address->GetDeviceType() != dest_device_address->GetDeviceType()) {
          MS_LOG_WITH_NODE(EXCEPTION, kernel_actor->kernel_)
            << "Invalid ref device address:" << ref_device_address << " type:" << ref_device_address->GetDeviceType()
            << " src device address:" << dest_device_address << " type:" << dest_device_address->GetDeviceType()
            << " for actor:" << GetAID();
        }
        MS_LOG(DEBUG) << "Add device tensor copy store for device address:" << ref_device_address
                      << " type:" << ref_device_address->GetDeviceType() << " and " << dest_device_address
                      << " type:" << dest_device_address->GetDeviceType() << " for actor:" << GetAID();
        DeviceTensorCopyStore::GetInstance().Insert(ref_device_address, dest_device_address);
        ref_device_address->set_ptr(dest_device_address->GetMutablePtr());
        MS_LOG(DEBUG) << "Set ptr:" << dest_device_address->GetMutablePtr()
                      << " from device address:" << dest_device_address << " to:" << ref_device_address
                      << " for actor:" << GetAID();
      }
    }
    // Update ref count.
    MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(src_device_address, src_device_context, GetAID().Name());
    MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(dest_device_address, dest_device_context, GetAID().Name());
  }

  return true;
}

void SuperKernelActor::UpdateOutputAddress(
  const std::vector<std::pair<size_t, std::vector<size_t>>> &kernel_inputs_to_actor_outputs,
  const KernelActorPtr &kernel_actor) {
  for (const auto &pair : kernel_inputs_to_actor_outputs) {
    size_t kernel_input_index = pair.first;
    DeviceTensor *real_input = kernel_actor->input_device_tensors_[kernel_input_index];
    MS_EXCEPTION_IF_NULL(real_input);

    const std::vector<size_t> &actor_output_indices = pair.second;
    for (auto actor_output_index : actor_output_indices) {
      auto data = output_data_[actor_output_index].first.get();
      MS_EXCEPTION_IF_NULL(data);
      data->data_ = real_input;
    }
  }
}

void SuperKernelActor::FetchParameterInput(const KernelActorPtr &kernel_actor, OpContext<DeviceTensor> *const context) {
  if (!enable_input_optimize_) {
    return;
  }
  auto graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  for (const auto &parameter_index : kernel_actor->parameter_indexs()) {
    auto device_tensor =
      FetchParameter(parameter_index.second, context, kernel_actor->device_contexts()[0], kernel_actor->GetAID());
    MS_LOG(DEBUG) << "Actor: " << kernel_actor->GetAID().Name() << ", input index: " << parameter_index.first
                  << ", device tensor: " << device_tensor << ", ptr: " << device_tensor->GetPtr()
                  << ", ref cnt: " << device_tensor->ref_count();
    // Device tensor in parameter store only keep one in each device, so the ref relation got lost
    // in multi heters with ref scenario.
    if (kernel_actor->modifiable_ref_input_indexes_.count(parameter_index.first) > 0) {
      auto outer_idx = parameter_index.second.second;
      auto inner_idx = parameter_index.second.first.second;
      graph_parameter_store->RefreshRefDeviceTensor(
        {{outer_idx, inner_idx}, kernel_actor->device_contexts()[0]->GetDeviceType()});
    }
    kernel_actor->SetInputDeviceTensor(device_tensor, parameter_index.first);
  }

  const auto &iter = kernel_actor_to_graph_parameters_map_.find(kernel_actor);
  if (iter != kernel_actor_to_graph_parameters_map_.end()) {
    for (const auto &input_pair : iter->second) {
      auto actor_input_idx = input_pair.first;
      auto graph_input_idx = input_pair.second;
      if (memory_free_lists_.empty()) {
        memory_free_lists_.push({});
      }
      CorrectRefCountByCondition(graph_input_idx, kernel_actor->input_device_tensors_[actor_input_idx],
                                 &memory_free_lists_.back());
      MS_LOG(DEBUG) << "Correct ref count for actor" << kernel_actor->GetAID().Name()
                    << ", actor input: " << actor_input_idx << ", graph input: " << graph_input_idx
                    << ", device tensor: " << kernel_actor->input_device_tensors_[actor_input_idx]
                    << ", ptr: " << kernel_actor->input_device_tensors_[actor_input_idx]->GetPtr()
                    << ", ref cnt: " << kernel_actor->input_device_tensors_[actor_input_idx]->ref_count();
    }
  }

  // Insert record wait pair to ensure first used parameter async copy end before launch.
  const auto &insert_event_iter = kernel_actors_insert_event_.find(kernel_actor.get());
  if (insert_event_iter != kernel_actors_insert_event_.end()) {
    auto stream_id = kernel_actor->kernel_info_->stream_id();
    if (stream_id != kDefaultStreamIndex) {
      auto multi_stream_controller = device::MultiStreamController::GetInstance();
      MS_EXCEPTION_IF_NULL(multi_stream_controller);
      auto device_context = kernel_actor->device_contexts_[0];
      MS_EXCEPTION_IF_NULL(device_context);
      MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
      device_context->device_res_manager_->BindDeviceToCurrentThread(false);
      multi_stream_controller->DispatchRecordWaitEvent(device_context, stream_id, kDefaultStreamIndex);
    }
  }

  for (const auto &parameter_index : kernel_actor->parameter_indexs()) {
    kernel_actor->memory_free_list_[parameter_index.first] = kernel_actor->input_device_tensors_[parameter_index.first];
    kernel_actor->CopyInputDeviceTensor(kernel_actor->input_device_tensors_[parameter_index.first],
                                        parameter_index.first, context);
  }
}

void SuperKernelActor::FreeInputParamWithoutUser(OpContext<DeviceTensor> *const context) {
  if (enable_input_optimize_) {
    for (const auto &iter : input_params_no_user_) {
      auto device_tensor = FetchParameter(iter.second, context, device_contexts_[0], GetAID());
      MS_EXCEPTION_IF_NULL(device_tensor);
      if (device_tensor->original_ref_count() != SIZE_MAX) {
        // No user for this input in graph.
        MemoryManagerActor::GetInstance()->FreeMemoryByRefCount(device_tensor, device_contexts_[0], GetAID().Name());
      }
    }
  }
}

bool SuperKernelActor::FetchMsgInputAndConstValueForKernel(KernelActor *kernel_actor,
                                                           OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  const auto &kernel = kernel_actor->kernel();

  // 1 Prepare received input from other actors.
  const auto &iter = kernel_input_to_graph_input_indices_.find(kernel.get());
  if (iter != kernel_input_to_graph_input_indices_.end()) {
    std::vector<std::pair<size_t, size_t>> &input_to_graph_input_indices = iter->second;
    for (const auto &item : input_to_graph_input_indices) {
      kernel_actor->SetInputDeviceTensor(input_device_tensors_[item.second], item.first);
      kernel_actor->memory_free_list_[item.first] = input_device_tensors_[item.second];
      kernel_actor->CopyInputDeviceTensor(input_device_tensors_[item.second], item.first, context);
    }
  }
  // 2. Prepare const value.
  if (!kernel_actor->device_tensor_store_keys_.empty()) {
    // Collect the inputs from device tensor store.
    kernel_actor->FetchInputByTensorStore(&kernel_actor->input_device_tensors_, &kernel_actor->input_kernel_tensors_,
                                          &kernel_actor->input_kernel_tensors_for_infer_,
                                          &kernel_actor->memory_free_list_, context);
    if (IsRunningFailed(context)) {
      return false;
    }
  }
  return true;
}

bool SuperKernelActor::LaunchAllKernels(OpContext<DeviceTensor> *const context) {
  size_t kernel_num = kernel_actors_.size();
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel_actor = kernel_actors_[i];
    if (kernel_actor == nullptr) {
      continue;
    }
    const auto &kernel = kernel_actor->kernel();
    if (device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, kernel_actor->GetAID().Name(),
                                                     kernel->fullname_with_scope(), kernel->func_graph()->ToString());
    }
    // 1. Prepare input data for kernel
    // 1.1. Prepare top cell parameter input.
    FetchParameterInput(kernel_actor, context);
    // 1.2. Prepare non top cell input, such as internal parameter msg input, control flow msg input and const value.
    if (!FetchMsgInputAndConstValueForKernel(kernel_actor.get(), context)) {
      return false;
    }

    // Update output device address for Parameter as graph output case.
    const auto &input_to_output_iter = kernel_input_to_actor_output_indices_.find(kernel.get());
    if (input_to_output_iter != kernel_input_to_actor_output_indices_.end()) {
      const auto &kernel_inputs_to_actor_outputs = input_to_output_iter->second;
      UpdateOutputAddress(kernel_inputs_to_actor_outputs, kernel_actor);
    }

    // 2. Allocate somas memory or cached memory for this kernel.
    kernel_actor->SetSomasMemory(context);
    if (ActorDispatcher::enable_use_trace_memory()) {
      SetTraceMemoryForKernel(kernel_actor);
    }

    // 3. Async Run Infer or Launch
    if (ActorDispatcher::enable_runtime_multi_pipeline() && !ActorDispatcher::enable_static_shape()) {
      // If the kernel need user data and is dynamic, maybe need input kernel's output user data to infer shape, this
      // value depend case can not handle in KernelTensor auto sync phase currently.
      if (kernel_actor->kernel_mod_->need_user_data() && kernel_actor->has_dynamic_) {
        MS_LOG(DEBUG) << "Begin wait runtime pipeline for kernel: " << kernel_actor->kernel_->fullname_with_scope();
        if (!WaitRuntimePipelineFinish(context)) {
          return false;
        }
        MS_LOG(DEBUG) << "End wait runtime pipeline for kernel: " << kernel_actor->kernel_->fullname_with_scope();
      }

      // Push run task to pipeline.
      // Note: dynamic value or static shape also need push task into infer actor to make sure correct kernel
      // execution order.
      Async(kernel_async_infer_aid_, &KernelAsyncInferActor::InferShape, context, kernel_actor.get());

      // The computed depend kernel should wait output shape update after kernel launch.
      if (kernel_actor->kernel_mod_->IsNeedUpdateOutputShapeAndSize()) {
        MS_LOG(DEBUG) << "Begin wait runtime pipeline for kernel: " << kernel_actor->kernel_->fullname_with_scope();
        if (!WaitRuntimePipelineFinish(context)) {
          return false;
        }
        MS_LOG(DEBUG) << "End wait runtime pipeline for kernel: " << kernel_actor->kernel_->fullname_with_scope();
      }
    } else {
      auto &llm_manager = LLMManager::GetInstance();
      if (llm_manager.need_force_resize(kernel_actor->kernel_mod_->kernel_name())) {
        kernel_actor->ResizeKernelMod();
        kernel_actor->FetchOutputDeviceTensor(context);
        kernel_actor->FetchWorkspaceDeviceTensor();
      } else if (!ActorDispatcher::enable_static_shape()) {
        kernel_actor->device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
        // Infer shape and resize for dynamic shape or dynamice value case when disable runtime multi pipeline.
        kernel_actor->InferAndUpdateDeviceTensorSize(context);
      }

      Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernel, context, kernel_actor.get());
    }

    // 4. Copy for heterogeneous output device address if need.
    if (kernel_actor->copy_output_device_tensors_.empty()) {
      continue;
    }
    if (!CopyHeterogeneousOutput(context, kernel_actor)) {
      MS_INTERNAL_EXCEPTION(RuntimeError)
        << "Copy for heterogeneous output failed, kernel actor: " << kernel_actor->GetAID().Name();
    }
  }

  return true;
}

void SuperKernelActor::DispatchParallelLaunchKernels(size_t index, OpContext<DeviceTensor> *const context) {
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
        MS_LOG(DEBUG) << "Begin launch kernel: " << kernel_actor->kernel_->fullname_with_scope();
        auto ret = device_contexts_[0]->GetKernelExecutor(false)->LaunchKernel(
          kernel_actor->kernel_, kernel_actor->input_kernel_tensors_, kernel_actor->workspace_kernel_tensors_,
          kernel_actor->output_kernel_tensors_, kernel_actor->kernel_mod_, real_stream);

        if (!ret) {
          MS_LOG(EXCEPTION) << "Launch kernel failed, kernel name: " << kernel_actor->kernel_->fullname_with_scope();
        }
        MS_LOG(DEBUG) << "End launch kernel: " << kernel_actor->kernel_->fullname_with_scope();
      }
    }

    events_[index + inner_index * parallel_dispatch_num_ + 1]->RecordEvent(real_stream_id);
  }
}

void SuperKernelActor::DispatchSerialLaunchKernels(OpContext<DeviceTensor> *const context) {
  auto *default_stream = device_contexts_[0]->device_res_manager_->GetStream(0);
  for (auto &kernel_actor : serial_launch_kernels_) {
    MS_EXCEPTION_IF_NULL(kernel_actor);
    auto comm_iter = serial_launch_kernels_to_events_.find(kernel_actor.get());
    if (comm_iter == serial_launch_kernels_to_events_.end()) {
      MS_LOG(EXCEPTION) << "Not find kernel actor : " << kernel_actor->kernel()->fullname_with_scope();
    }

    const auto &kernel = kernel_actor->kernel_;
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

    auto &llm_manager = LLMManager::GetInstance();
    bool need_force_resize = llm_manager.need_force_resize(kernel_actor->kernel_mod_->kernel_name());
    if (need_force_resize) {
      kernel_actor->ResizeKernelMod();
      kernel_actor->FetchOutputDeviceTensor(nullptr);
      kernel_actor->FetchWorkspaceDeviceTensor();
    }

    const auto &event_array = comm_iter->second;
    auto &wait_event = event_array[0];
    wait_event->WaitEventWithoutReset(0);

    MS_LOG(DEBUG) << "Begin serial launch kernel: " << kernel_actor->kernel_->fullname_with_scope();
    auto ret = device_contexts_[0]->GetKernelExecutor(false)->LaunchKernel(
      kernel_actor->kernel_, kernel_actor->input_kernel_tensors_, kernel_actor->workspace_kernel_tensors_,
      kernel_actor->output_kernel_tensors_, kernel_actor->kernel_mod_, default_stream);
    if (!ret) {
      MS_LOG(EXCEPTION) << "Launch kernel failed, kernel name: " << kernel_actor->kernel_->fullname_with_scope();
    }
    MS_LOG(DEBUG) << "End serial launch kernel: " << kernel_actor->kernel_->fullname_with_scope();

    auto &record_event = event_array[1];
    record_event->RecordEvent(0);
  }
}

void SuperKernelActor::ParallelDispatchKernels(OpContext<DeviceTensor> *const context) {
  MS_LOG(INFO) << "Begin parallel dispatch kernels for graph: " << graph_->ToString();
  device_contexts_[0]->device_res_manager_->BindDeviceToCurrentThread(false);
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
  MS_LOG(INFO) << "End parallel dispatch kernels for graph: " << graph_->ToString();
}

void SuperKernelActor::RunGraphKernelByKernel(OpContext<DeviceTensor> *const context) {
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

  // 1. Fetch input data for this kernel graph and correct current ref count for input device address.
  FetchInputDeviceTensor(context);
  FreeInputParamWithoutUser(context);

  // 2. Allocate somas memory for graph
  if ((somas_info_ != nullptr) && (somas_info_->whole_block_size_ != 0)) {
    MemoryManagerActor::GetInstance()->AllocateSomasMemory(somas_info_, device_contexts_[0], context, GetAID());
  }

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

  if (ActorDispatcher::enable_parallel_dispatch_kernel_for_cur_actor_set()) {
    ParallelDispatchKernels(context);
  } else {
    // 3. Launch all kernels by execution order in kernel graph.
    if (!LaunchAllKernels(context)) {
      MS_INTERNAL_EXCEPTION(RuntimeError)
        << "Launch kernels by execution order failed for graph: " << graph_->ToString();
    }
  }

  if (((somas_info_ != nullptr) && (somas_info_->whole_block_size_ != 0)) ||
      ActorDispatcher::enable_trace_dynamic_memory() || ActorDispatcher::enable_use_trace_memory()) {
    WaitRuntimePipelineFinish(context);
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
    FreeTraceMemory();
  }

  // Free input data for use trace memory (run by static shape) step.
  PostRun(context);
}

void SuperKernelActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                  "Invalid device context for super kernel actor:" + GetAID().Name());
  }
  sort(memory_alloc_list_.begin(), memory_alloc_list_.end(), [](const DeviceTensor *a, const DeviceTensor *b) {
    MS_EXCEPTION_IF_NULL(a);
    MS_EXCEPTION_IF_NULL(b);
    return a->GetSize() > b->GetSize();
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

void SuperKernelActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
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
    const std::vector<tensor::TensorPtr> inputs;
    std::vector<tensor::TensorPtr> outputs;
    const std::map<string, string> compile_options;
    if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                    "Invalid device context for super kernel actor:" + GetAID().Name());
    }
    MS_EXCEPTION_IF_NULL(device_contexts_[0]->graph_executor_);
    if (!IsSkippedLaunch(nullptr, graph_)) {
      ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kGraphLaunch, GetAID().Name());
      auto ret = device_contexts_[0]->graph_executor_->RunGraph(graph_, inputs, &outputs, compile_options);
      if (!ret) {
        std::string error_info = "Launch graph failed, graph id: " + std::to_string(graph_->graph_id());
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
    } else if (IsNeedProfilieMemoryLog()) {
      auto memory_size = device_contexts_[0]->graph_executor_->GetGraphFeatureMemory(graph_);
      MS_LOG(WARNING) << "Need Profile Memory, graph: " << graph_->ToString() << ", feature memory: " << memory_size;
      MS_LOG(WARNING) << "Need Profile Memory, max used static memory: "
                      << device_contexts_[0]->device_res_manager_->GetMaxUsedMemorySize();
    }
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
      MS_LOG(INFO) << "The input ref node copy back from address: " << item.first->GetPtr()
                   << " to address: " << item.second->GetPtr() << ".";
      if (!Copy(item.second, item.first)) {
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

void SuperKernelActor::SendDebugReq(OpContext<DeviceTensor> *const context) {
  running_dependent_msg_num_ = 1;
  if (device_contexts_.empty() || device_contexts_[0] == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                  "Invalid device context for super kernel actor:" + GetAID().Name());
  }
  OnDebugFinish(context);
}

bool SuperKernelActor::CopyInputDataPersistedHandle(const DeviceContext *device_context,
                                                    DeviceTensor *input_device_tensor,
                                                    const DeviceTensorPtr &node_device_tensor, size_t i) {
  if ((input_device_tensor->GetDeviceType() == node_device_tensor->GetDeviceType()) &&
      AnfAlgo::IsEquivalentFormat(input_device_tensor->format(), node_device_tensor->format())) {
    MS_LOG(DEBUG) << "Not need copy for device tensor:" << node_device_tensor << " ptr:" << node_device_tensor->GetPtr()
                  << " index:" << i << " for actor:" << GetAID();
    // Set the ptr from input_device_tensor and set mem pool false to avoid memory double management for
    // supporting zero copy.
    if (type_ != KernelTransformType::kSuperKernelActor) {
      node_device_tensor->set_ptr(input_device_tensor->GetMutablePtr());
    } else {
      node_device_tensor->set_ptr(input_device_tensor->GetValidPtr(input_device_tensor->stream_id()));
    }
    MS_LOG(DEBUG) << "Actor:" << GetAID() << "set need sync flag from:" << input_device_tensor
                  << " to:" << node_device_tensor
                  << " sync user data handler:" << node_device_tensor->need_sync_user_data();
    node_device_tensor->set_from_mem_pool(false);
    // continue
    return true;
  }
  if (device_context->GetDeviceType() != node_device_tensor->GetDeviceType()) {
    device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {node_device_tensor->device_name(), node_device_tensor->device_id()});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->device_res_manager_);
  }

  if (copy_input_device_tensors_[i] == nullptr) {
    MS_EXCEPTION_IF_NULL(node_device_tensor->kernel_tensor());
    const auto new_kernel_tensor = node_device_tensor->kernel_tensor()->CloneKernelTensor();
    MS_EXCEPTION_IF_NULL(new_kernel_tensor);
    new_kernel_tensor->set_device_name(device_context->device_context_key().device_name_);
    new_kernel_tensor->set_device_id(device_context->device_context_key().device_id_);
    new_kernel_tensor->set_device_ptr(nullptr);

    copy_input_device_tensors_[i] = device_context->device_res_manager_->CreateDeviceAddress(new_kernel_tensor);
    MS_LOG(DEBUG) << "Create new device tensor:" << copy_input_device_tensors_[i] << " index:" << i
                  << " for actor:" << GetAID();
  }
  auto copy_device_tensor = copy_input_device_tensors_[i];
  MS_EXCEPTION_IF_NULL(copy_device_tensor);
  copy_device_tensor->set_user_data(node_device_tensor->user_data());
  copy_device_tensor->set_need_sync_user_data(node_device_tensor->need_sync_user_data());
  if ((copy_device_tensor->GetPtr() == nullptr) &&
      (!device_context->device_res_manager_->AllocateMemory(copy_device_tensor.get()))) {
    MS_LOG(ERROR) << "Device(id:" << std::to_string(device_context->device_context_key().device_id_)
                  << ") memory isn't enough and alloc failed, kernel name: " << GetAID()
                  << ", alloc size: " + std::to_string(copy_device_tensor->GetSize()) << "B.";
    return true;
  }
  MS_LOG(DEBUG) << "Alloc memory for device tensor:" << copy_device_tensor << " ptr:" << copy_device_tensor->GetPtr()
                << " size:" << copy_device_tensor->GetSize() << " index:" << i << " for actor:" << GetAID();
  if (type_ != KernelTransformType::kSuperKernelActor) {
    node_device_tensor->set_ptr(copy_device_tensor->GetMutablePtr());
  } else {
    node_device_tensor->set_ptr(copy_device_tensor->GetValidPtr(copy_device_tensor->stream_id()));
  }
  node_device_tensor->set_from_mem_pool(false);
  return false;
}

bool SuperKernelActor::CopyInputData(const OpContext<DeviceTensor> *context, const KernelGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph);
  if (device_contexts_.empty() || device_contexts_[0] == nullptr ||
      device_contexts_[0]->device_res_manager_ == nullptr) {
    MS_LOG(ERROR) << "Invalid device context for actor:" << GetAID();
    return false;
  }
  auto device_context = device_contexts_[0];
  auto &input_nodes = graph->input_nodes();
  if (input_device_tensors_.size() != node_device_tensors_.size()) {
    MS_LOG(ERROR) << "The size of input_device_tensors_[" << input_device_tensors_.size()
                  << "] is not equal to the size of node_device_tensors_[" << node_device_tensors_.size() << "].";
    return false;
  }

  for (size_t i = 0; i < input_device_tensors_.size(); ++i) {
    auto &node_device_tensor = node_device_tensors_[i];
    MS_EXCEPTION_IF_NULL(node_device_tensor);
    auto &input_device_tensor = input_device_tensors_[i];
    if (InputDataNoNeedCopy(input_nodes[i], input_device_tensor, node_device_tensor, type_)) {
      MS_LOG(DEBUG) << "Actor:" << GetAID() << " input device tensor " << i << ":" << input_device_tensor
                    << " no need copy.";
      continue;
    }
    MS_EXCEPTION_IF_NULL(input_nodes[i]);
    const auto &node_device_kernel_tensor = node_device_tensor->kernel_tensor();
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    const auto &input_kernel_tensor = input_device_tensor->kernel_tensor();
    MS_EXCEPTION_IF_NULL(node_device_kernel_tensor);
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    UpdateShape(input_nodes[i], node_device_tensor, input_device_tensor, type_);
    node_device_tensor->set_user_data(input_device_tensor->user_data());
    node_device_tensor->set_need_sync_user_data(input_device_tensor->need_sync_user_data());
    if (type_ != KernelTransformType::kSuperKernelActor) {
      node_device_kernel_tensor->SetValue(input_kernel_tensor->GetValueTrack());
    }

    // Copy.
    DeviceTensorPtr copy_device_tensor = nullptr;
    // If the input is not a persist device address, in a heterogeneous scenario, a new device address needs to
    // be created. And set ptr to node device address to support the zero copy of graph input nodes.
    if (!node_device_tensor->is_ptr_persisted()) {
      if (CopyInputDataPersistedHandle(device_context, input_device_tensor, node_device_tensor, i)) {
        continue;
      }
      copy_device_tensor = copy_input_device_tensors_[i];
    } else {
      if (node_device_tensor->GetPtr() == nullptr) {
        MS_LOG(INFO) << "The node device tensor, which shared with another graph, has no device memory and will skip "
                        "copy for actor:"
                     << GetAID();
        continue;
      }
      copy_device_tensor = node_device_tensor;
    }
    MS_EXCEPTION_IF_NULL(copy_device_tensor);
    MS_LOG(INFO) << "The input data of node:" << input_nodes[i]->DebugString()
                 << " need copy from device address:" << input_device_tensor << " ptr:" << input_device_tensor->GetPtr()
                 << " size:" << input_device_tensor->GetSize() << ", type:" << input_device_tensor->GetDeviceType()
                 << " to device address:" << copy_device_tensor << " ptr:" << copy_device_tensor->GetPtr()
                 << " size:" << copy_device_tensor->GetSize() << ", type:" << copy_device_tensor->GetDeviceType()
                 << ", is ref node need copy back:" << is_parameters_need_copy_[i] << " for actor:" << GetAID();
    if (!Copy(copy_device_tensor.get(), input_device_tensor)) {
      MS_LOG(ERROR) << "Copy data failed for actor:" << GetAID() << " input index:" << i;
      continue;
    }

    if (is_parameters_need_copy_[i]) {
      ref_node_addr_map_[copy_device_tensor.get()] = input_device_tensor;
    }
  }
  return true;
}

void SuperKernelActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(graph_);

  if (device_contexts_.empty() || device_contexts_[0] == nullptr ||
      device_contexts_[0]->device_res_manager_ == nullptr) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context),
                                                  "Invalid device context for super kernel actor:" + GetAID().Name());
  }
  if (memory_free_lists_.size() > 0 && memory_free_lists_.back().size() > 0) {
    if (IsNeedProfilieMemoryLog()) {
      for (auto data : memory_free_lists_.back()) {
        auto output_address = reinterpret_cast<std::uintptr_t>(data);
        MS_LOG(WARNING) << "Need Profile Memory, Memory need Decrease DynamicRefCount, actor name: " << GetAID().Name()
                        << ", kernel graph: " << graph_->ToString() << ", device address class ptr: " << output_address
                        << ", device address size: " << data->GetSize() << ", device address addr: " << data->GetPtr();
      }
    }

    if (ActorDispatcher::is_memory_free_sync()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                                device_contexts_[0], context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &(memory_free_lists_.back()),
                            device_contexts_[0], context, GetAID());
    }
  }

  // Free the address that is the temp store for kernel input copy.
  for (auto &copy_input_device_tensor : copy_input_device_tensors_) {
    if ((copy_input_device_tensor != nullptr) && (copy_input_device_tensor->GetPtr() != nullptr)) {
      device_contexts_[0]->device_res_manager_->FreeMemory(copy_input_device_tensor.get());
    }
  }
}

void SuperKernelActor::BuildAndLinkKernelActors() {
  if (enable_kbk_sub_graph_execute_) {
    BuildKernelActors();
    LinkKernelActors();

    if (enable_parallel_dispatch_) {
      PartitionParallelDispatchKernels();
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
      parallel_launch_kernels_[i] = std::vector<KernelActorPtr>(begin_iter + i * kernel_num_per_dispatcher,
                                                                begin_iter + (i + 1) * kernel_num_per_dispatcher);
    } else {
      parallel_launch_kernels_[i] =
        std::vector<KernelActorPtr>(begin_iter + i * kernel_num_per_dispatcher, kernel_actors_.end());
    }
    MS_LOG(INFO) << "The kernel group[" << i << "] kernel num: " << parallel_launch_kernels_[i].size();
  }

  // Get serial launch kernels.
  for (auto &kernel_actor : kernel_actors_) {
    if (!kernel_actor) {
      continue;
    }
    auto &llm_manager = LLMManager::GetInstance();
    const auto &kernel_name = kernel_actor->kernel_mod_->kernel_name();
    bool need_force_resize = llm_manager.need_force_resize(kernel_name);
    if (need_force_resize || (common::AnfAlgo::IsCommunicationOp(kernel_actor->kernel_) ||
                              kernel_name == "QbmmAllReduceAdd" || kernel_name == "MatmulAllReduceAddRmsNorm")) {
      serial_launch_kernels_.push_back(kernel_actor);
    } else if (kernel_name.find(kAllReduceOpName) != std::string::npos ||
               kernel_name.find(kAllGatherOpName) != std::string::npos ||
               kernel_name.find(kReduceScatterOpName) != std::string::npos ||
               kernel_name.find(kAllToAllOpName) != std::string::npos ||
               kernel_name.find(kAlltoAllOpName) != std::string::npos) {
      MS_LOG(WARNING) << "Find parallel dispatch communication op: " << kernel_name;
    }
  }
}

void SuperKernelActor::BuildKernelActors() {
  MS_EXCEPTION_IF_NULL(graph_);
  const auto &execution_order = graph_->execution_order();
  size_t kernel_num = execution_order.size();
  kernel_actors_.resize(kernel_num);

  mindspore::HashMap<uint32_t, std::pair<KernelActorPtr, KernelActorPtr>> send_recv_nodes;
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
    const auto &real_device_context = device::FetchRealDeviceContext(kernel, device_contexts_[0]);
    MS_EXCEPTION_IF_NULL(real_device_context);
    if (IsRpcActor(kernel)) {
      MS_LOG(EXCEPTION) << "Can not launch a sub graph which contains rpc kernel by kbk.";
    } else if (IsInnerControlFlowActor(kernel)) {
      MS_LOG(EXCEPTION) << "Can not launch a sub graph which contains ConditionSwitch or ConditionSwitch by kbk.";
    }

    KernelActorPtr kernel_actor = std::make_shared<KernelActor>(
      GenerateActorIdByKernel(kernel), kernel, real_device_context, memory_manager_aid_, debug_aid_, recorder_aid_,
      GraphExecutionStrategy::kPipeline, ref_input_indexes, ref_output_indexes);
    MS_EXCEPTION_IF_NULL(kernel_actor);
    kernel_actors_[i] = kernel_actor;

    // Set the member of kernel actor.
    kernel_actor->is_launch_skipped_ =
      common::AnfAlgo::IsNopNode(kernel) && graph_->IsInRefOutputMap(std::make_pair(kernel, 0));
    kernel_actor->inputs_continuous_memory_ =
      (common::AnfAlgo::IsCommunicationOp(kernel) && common::AnfAlgo::GetCNodeName(kernel) != kMatMulAllReduceOpName) &&
      (common::AnfAlgo::GetInputTensorNum(kernel) > 1);

    if (SchedulerHelper::IsSkipLaunchShapeRelatedOp(kernel_actor.get())) {
      kernel_actor->set_skip_launch_shape_related_op(true);
    }

    if (IsPrimitiveCNode(kernel, prim::kPrimStreamSend)) {
      SchedulerHelper::ProcessStreamSendRecvEventPair(&send_recv_nodes, kernel, kernel_actor, true);
    } else if (IsPrimitiveCNode(kernel, prim::kPrimStreamRecv)) {
      SchedulerHelper::ProcessStreamSendRecvEventPair(&send_recv_nodes, kernel, kernel_actor, false);
    }

    SchedulerHelper::AddSomasInfo(kernel_actor.get());

    cnode_to_kernel_actor_[kernel] = kernel_actor.get();
  }
  for (auto &[event_pair_id, send_recv_actor] : send_recv_nodes) {
    auto [send_actor, recv_actor] = send_recv_actor;
    MS_LOG(DEBUG) << "Stream send/recv pair : " << event_pair_id << ", send_actor : " << send_actor
                  << ", recv_actor : " << recv_actor << ".";
    recv_actor->set_stream_send_actor(send_actor.get());
  }

  // 2. Add somas info.
  // AddSomasOutput
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
    output_actor->is_output_kernel_ = true;
    SchedulerHelper::AddSomasInfoForGraphOutput(output_actor, output_index, graph_->graph_id());
  }

  // 3. Initialize all kernel actor.
  // Note: this step must execute before LinkKernelActors, LinkKernelActors will check whether the output ref count is
  // max or not to optimize free performance for somas case, need not try to free the output which has a max ref count.
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel_actor = kernel_actors_[i];
    if (kernel_actor) {
      kernel_actor->Init();
    }
  }
}

void SuperKernelActor::LinkKernelActors() {
  const auto &input_nodes = graph_->input_nodes();
  size_t input_num = input_nodes.size();
  param_node_to_input_idx_.reserve(input_num);
  // Record the parameter first used actor and actor input idx.
  std::vector<std::pair<KernelActorPtr, size_t>> param_first_used_kernel_actors(input_num, {nullptr, 0});
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

  // 3. Check input parameter(not persist tensor) as graph output case, need increase the parameter use count for output
  // parameter.
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

  if (IS_OUTPUT_ON(MsLogLevel::kDebug)) {
    for (size_t i = 0; i < input_num; i++) {
      MS_LOG(DEBUG) << "SuperKernelActor: " << GetAID().Name() << " Parameter[" << input_nodes[i]->fullname_with_scope()
                    << "] debug_name: " << input_nodes[i]->DebugString()
                    << " use count is: " << input_params_use_cnt_[i];
    }
  }
}

void SuperKernelActor::AnalyseNodesDependence(
  const HashMap<size_t, AnfNodePtr> &device_tensor_store_keys_map,
  const HashMap<size_t, ParameterInfo> &parameter_indexs_map,
  const HashMap<AnfNodePtr, std::vector<size_t>> &output_node_to_actor_output_index,
  std::vector<std::pair<KernelActorPtr, size_t>> *param_first_used_kernel_actors) {
  const auto &execution_order = graph_->execution_order();
  mindspore::HashMap<size_t, mindspore::HashMap<size_t, KernelActorPtr>> param_first_used_actors_on_stream;
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
          (void)kernel_actor->device_tensor_store_keys_.emplace_back(j, device_tensor_store_key_iter->second);
        }

        const auto &parameter_index_iter = parameter_indexs_map.find(input_node_idx);
        if (parameter_index_iter != parameter_indexs_map.end()) {
          auto &kernel_actor = kernel_actors_[i];
          MS_EXCEPTION_IF_NULL(kernel_actor);
          (void)kernel_actor->parameter_indexs_.emplace_back(j, parameter_index_iter->second);
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
        }
      }
    }
  }
  CollectStreamFirstUsedParamKernelActors(&param_first_used_actors_on_stream, &kernel_actors_insert_event_);
  ParamFirstUsedKernelActorsToMap(*param_first_used_kernel_actors, &kernel_actor_to_graph_parameters_map_);
  RecordInputParamsWithoutUser(graph_, parameter_indexs_map, input_params_use_cnt_, &input_params_no_user_);
}

void SuperKernelActor::LinkKernelActor(const CNodePtr &kernel, size_t input_index, const AnfNodePtr &input_kernel,
                                       size_t output_index) {
  // Shape depend kernel should not increase ref count.
  if (IsOnlyDependShape(kernel, input_index)) {
    auto device_tensor = AnfAlgo::GetMutableOutputAddr(input_kernel, output_index, false);
    MS_EXCEPTION_IF_NULL(device_tensor);
    device_tensor->UpdateFlag(device::kDeviceAddressFlagNullptr);

    auto *kernel_actor = cnode_to_kernel_actor_[kernel];
    MS_EXCEPTION_IF_NULL(kernel_actor);
    kernel_actor->SetInputDeviceTensor(device_tensor.get(), input_index);
    kernel_actor->memory_free_list_[input_index] = device_tensor.get();
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

  const auto &input_device_tensor = AnfAlgo::GetMutableOutputAddr(input_kernel, output_index, false);
  MS_EXCEPTION_IF_NULL(input_device_tensor);

  bool need_not_copy_output_device_addr = (device_context->GetDeviceType() == input_device_context->GetDeviceType()) ||
                                          SchedulerHelper::IsIgnoredInputAddress(kernel_actor, input_index);
  MS_LOG(DEBUG) << "Kernel:" << kernel->fullname_with_scope() << " input kernel:" << input_kernel->fullname_with_scope()
                << " need copy:" << need_not_copy_output_device_addr << " for actor:" << GetAID();
  if (need_not_copy_output_device_addr) {
    UpdateRefCount(input_device_tensor.get(), false);
    kernel_actor->SetInputDeviceTensor(input_device_tensor.get(), input_index);
    kernel_actor->memory_free_list_[input_index] = input_device_tensor.get();
    return;
  }

  auto &copy_output_device_tensors = input_kernel_actor->copy_output_device_tensors_;
  auto iter = copy_output_device_tensors.find(output_index);
  if (iter == copy_output_device_tensors.end()) {
    const auto &input_kernel_tensor = input_device_tensor->kernel_tensor();
    const auto input_copy_kernel_tensor = input_kernel_tensor->CloneKernelTensor();
    MS_EXCEPTION_IF_NULL(input_copy_kernel_tensor);
    input_copy_kernel_tensor->set_device_name(device_context->device_context_key().device_name_);
    input_copy_kernel_tensor->set_device_id(device_context->device_context_key().device_id_);
    input_copy_kernel_tensor->set_device_ptr(nullptr);

    auto input_copy_device_address = device_context->device_res_manager_->CreateDeviceAddress(input_copy_kernel_tensor);
    auto ret_pair = copy_output_device_tensors.emplace(
      output_index,
      std::make_pair(input_copy_device_address, std::make_pair(device_context, std::vector<DeviceTensor *>())));
    if (ret_pair.second) {
      iter = ret_pair.first;
    } else {
      MS_LOG(EXCEPTION) << "Insert copy output device address failed.";
    }
    UpdateRefCount(input_device_tensor.get(), false);
  }

  const auto &input_copy_device_address = iter->second.first;
  MS_EXCEPTION_IF_NULL(input_copy_device_address);
  UpdateRefCount(input_copy_device_address.get(), false);
  if (kernel_actor->modifiable_ref_input_indexes_.count(input_index) > 0) {
    MS_LOG(DEBUG) << "Add device tensor copy store for device address:" << input_copy_device_address
                  << " type:" << input_copy_device_address->GetDeviceType() << " and " << input_device_tensor
                  << " type:" << input_device_tensor->GetDeviceType() << " for copy actor:" << GetAID();
    DeviceTensorCopyStore::GetInstance().Insert(input_copy_device_address.get(), input_device_tensor.get());
    if (kernel_actor->kernel_info_ != nullptr) {
      const auto &ref_map = kernel_actor->kernel_info_->out_in_ref_map();
      auto index_iter =
        std::find_if(ref_map.begin(), ref_map.end(),
                     [input_index](const std::pair<size_t, size_t> &pair) { return pair.second == input_index; });
      if (index_iter != ref_map.end() && kernel_actor->output_device_tensors_.size() > index_iter->first &&
          kernel_actor->output_device_tensors_[index_iter->first] != nullptr) {
        UpdateRefCount(input_copy_device_address.get(), true);
        iter->second.second.second.emplace_back(kernel_actor->output_device_tensors_[index_iter->first]);
        MS_LOG(DEBUG) << "Add dst device address:" << kernel_actor->output_device_tensors_[index_iter->first]
                      << " for input copy device address:" << input_copy_device_address
                      << " for actor:" << kernel_actor->GetAID();
      }
    }
  }
  kernel_actor->SetInputDeviceTensor(input_copy_device_address.get(), input_index);
  kernel_actor->memory_free_list_[input_index] = input_copy_device_address.get();
}

void SuperKernelActor::TrackInputMemory() {
  if (!device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    return;
  }

  for (auto &device_addr : input_device_tensors_) {
    if (device_addr == nullptr || !device_addr->IsPtrValid()) {
      continue;
    }
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(UseMemBlock, GetAID().Name(), device_addr->GetPtr());
  }
}
}  // namespace runtime
}  // namespace mindspore
