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
#include "async/async.h"
#include "utils/phase.h"
#include "utils/llm_manager.h"
#include "utils/log_adapter.h"
#include "op_def/framework_ops.h"

namespace mindspore {
namespace runtime {
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
}  // namespace
void SuperKernelActor::Init() {
  MS_EXCEPTION_IF_NULL(graph_);
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  // Set the number of actor running dependent messages.
  running_dependent_msg_num_ = SizeToInt(input_datas_num_ + input_controls_num_);

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
      if (common::IsNeedProfileMemory()) {
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

      if (common::IsNeedProfileMemory()) {
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

      // There is no memory free action for use trace memory step, need to free input device address of the kernel graph
      // after launch all kernels.
      if (ActorDispatcher::enable_use_trace_memory()) {
        if (input_data->data_->original_ref_count() != SIZE_MAX ||
            input_data->data_->dynamic_ref_count() != INT32_MAX) {
          (void)memory_free_list.emplace_back(input_data->data_);
        }
      } else {
        CorrectRefCount(index, input_data->data_);
      }
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
  if (common::IsNeedProfileMemory()) {
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
    already_fetch_persistent_device_tensor_ = IsTwoPhaseInfer();
  }

  TrackInputMemory();

  if (memory_alloc_list_.size() > 0) {
    for (auto &device_tensor : memory_alloc_list_) {
      MS_EXCEPTION_IF_NULL(device_tensor);
      if (device_tensor->IsNotNeedAlloc()) {
        continue;
      }
      if (common::IsNeedProfileMemory()) {
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
  if (common::IsNeedProfileMemory()) {
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

void SuperKernelActor::SetTraceMemoryForKernel(const KernelActorPtr &kernel_actor) {
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
        kernel_actor->output_kernel_tensors_.at(block->index_)->set_device_ptr(ptr);
      } else {
        kernel_actor->workspace_kernel_tensors_.at(block->index_)->set_device_ptr(ptr);
      }
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

  for (const auto &output_index_to_copy_address : kernel_actor->copy_output_device_tensors_) {
    const auto &output_index = output_index_to_copy_address.first;
    const auto &dest_device_address = output_index_to_copy_address.second.first.get();
    const auto &dest_device_context = output_index_to_copy_address.second.second;
    const auto &src_device_address = kernel_actor->output_device_tensors_.at(output_index);
    const auto &src_device_context = kernel_actor->device_contexts_[0];

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
      MS_LOG_WITH_NODE(EXCEPTION, kernel_actor->kernel_)
        << "Memory leak detected in copy output device address for kernel: "
        << kernel_actor->kernel_->fullname_with_scope();
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
    const auto &iter = kernel_input_to_graph_input_indices_.find(kernel.get());
    if (iter != kernel_input_to_graph_input_indices_.end()) {
      std::vector<std::pair<size_t, size_t>> &input_to_graph_input_indices = iter->second;
      for (const auto &item : input_to_graph_input_indices) {
        kernel_actor->SetInputDeviceTensor(input_device_tensors_[item.second], item.first);
        kernel_actor->memory_free_list_[item.first] = input_device_tensors_[item.second];
        kernel_actor->CopyInputDeviceTensor(input_device_tensors_[item.second], item.first, context);
      }
    }
    if (!kernel_actor->device_tensor_store_keys_.empty()) {
      // Collect the inputs from device tensor store.
      kernel_actor->FetchInputByTensorStore(&kernel_actor->input_device_tensors_, &kernel_actor->input_kernel_tensors_,
                                            &kernel_actor->input_kernel_tensors_for_infer_,
                                            &kernel_actor->memory_free_list_, context);
      if (IsRunningFailed(context)) {
        return false;
      }
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

void SuperKernelActor::RunGraphKernelByKernel(OpContext<DeviceTensor> *const context) {
  // Mode check for dynamic shape, async launch and runtime multi pipeline.
  if (!ActorDispatcher::enable_async_launch_kernel()) {
    std::string error_info =
      "Runtime pipeline optimization is disabled, failed to execute graph kernel by kernel mode.";
    MS_LOG(ERROR) << "Run graph failed, graph id: " << std::to_string(graph_->graph_id()) << ". " << error_info;
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }
  if (graph_->is_dynamic_shape() && !ActorDispatcher::enable_runtime_multi_pipeline()) {
    MS_INTERNAL_EXCEPTION(RuntimeError) << "Must enable runtime multi pipeline when run dynamic shape graph.";
  }
  if (!graph_->is_dynamic_shape()) {
    ActorDispatcher::set_enable_static_shape(false);
  }

  // 1. Fetch input data for this kernel graph and correct current ref count for input device address.
  FetchInputDeviceTensor(context);

  // 2. Allocate somas memory for graph
  if ((somas_info_ != nullptr) && (somas_info_->whole_block_size_ != 0)) {
    MemoryManagerActor::GetInstance()->AllocateSomasMemory(somas_info_, device_contexts_[0], context, GetAID());
  }
  const auto &phase = PhaseManager::GetInstance().phase();
  bool is_increment_graph = (phase.find("increment") != std::string::npos);
  if (enable_trace_memory_ && graph_->is_dynamic_shape() && is_increment_graph) {
    MS_LOG(DEBUG) << "Enable trace memory for increment inference graph: " << graph_->graph_id()
                  << ", phase: " << phase;
    UpdateMemoryTraceMangerStatus(context);
    if (IsRunningFailed(context)) {
      // Maybe allocate memory failed, early stop to run graph.
      MS_LOG(INFO) << "Run failed and early stop to run graph: " << graph_->ToString();
      return;
    }
  }

  // 3. Launch all kernels by execution order in kernel graph.
  if (!LaunchAllKernels(context)) {
    MS_INTERNAL_EXCEPTION(RuntimeError) << "Launch kernels by execution order failed for graph: " << graph_->ToString();
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
    const std::vector<tensor::Tensor> inputs;
    std::vector<tensor::Tensor> outputs;
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
    } else if (common::IsNeedProfileMemory()) {
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
    if (common::IsNeedProfileMemory()) {
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
      kernel->fullname_with_scope(), kernel, real_device_context, memory_manager_aid_, debug_aid_, recorder_aid_,
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
  AnalyseNodesDependence(device_tensor_store_keys_map, output_node_to_actor_output_index);

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
  const HashMap<AnfNodePtr, std::vector<size_t>> &output_node_to_actor_output_index) {
  const auto &execution_order = graph_->execution_order();
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
        } else {
          kernel_input_to_graph_input_indices_[kernel.get()].emplace_back(j, input_node_idx);
        }

        auto output_idx_iter = output_node_to_actor_output_index.find(input_node_with_idx.first);
        if (output_idx_iter != output_node_to_actor_output_index.end()) {
          kernel_input_to_actor_output_indices_[kernel.get()].emplace_back(j, output_idx_iter->second);
        }
      }
    }
  }
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
    auto ret_pair =
      copy_output_device_tensors.emplace(output_index, std::make_pair(input_copy_device_address, device_context));
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
