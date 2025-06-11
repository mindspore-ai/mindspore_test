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

#include <string>
#include "runtime/graph_scheduler/graph_capture/graph_capture_manager.h"
#include "include/common/runtime_conf/runtime_conf.h"
#include "runtime/graph_scheduler/actor/super_kernel_actor.h"
#include "utils/llm_manager.h"

namespace mindspore {
namespace runtime {
GraphCaptureManager &GraphCaptureManager::GetInstance() noexcept {
  static GraphCaptureManager instance{};
  return instance;
}

bool GraphCaptureManager::GetEnableGraphCapture() const {
  auto runtime_conf_instance = runtime::RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  return runtime_conf_instance->GetEnableKernelLaunchCapture();
}

void GraphCaptureManager::SetEnableGraphCapture(bool enable_graph_capture) {
  auto runtime_conf_instance = runtime::RuntimeConf::GetInstance();
  MS_EXCEPTION_IF_NULL(runtime_conf_instance);
  runtime_conf_instance->SetKernelLaunchCapture(enable_graph_capture);
}

bool GraphCaptureManager::CheckKernelSupportCapture(const KernelRunnerPtr &kernel_runner,
                                                    const DeviceContext *expected_device_context) {
  MS_EXCEPTION_IF_NULL(kernel_runner);
  const auto &kernel = kernel_runner->kernel();
  MS_EXCEPTION_IF_NULL(kernel);
  const auto &kernel_mod = kernel_runner->kernel_mod();
  MS_EXCEPTION_IF_NULL(kernel_mod);

  auto kernel_type = AnfAlgo::GetKernelType(kernel);
  if (kernel_type == KernelType::ACL_KERNEL) {
    return false;
  }

  if ((kernel_runner->device_contexts())[0]->GetDeviceType() != expected_device_context->GetDeviceType()) {
    MS_LOG(EXCEPTION) << "Capture graph mode can not support cpu kernel: " << kernel->fullname_with_scope();
  }

  if (kernel_mod->IsNeedUpdateOutputShapeAndSize()) {
    MS_LOG(EXCEPTION)
      << "Capture graph mode can not support computed depend kernel(whose shape need update after launch.): "
      << kernel->fullname_with_scope();
  }

  auto &llm_manager = LLMManager::GetInstance();
  if (llm_manager.need_force_resize(kernel_mod->kernel_name())) {
    return false;
  }

  return true;
}

bool GraphCaptureManager::FindSupportCaptureKernelPositions(const std::vector<KernelRunnerPtr> &kernel_runners,
                                                            const DeviceContext *expected_device_context) {
  size_t start = 0;
  size_t end = 0;
  bool find_kernel_can_capture = false;
  size_t kernel_num = kernel_runners.size();
  if (kernel_num < 1) {
    return false;
  }
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel_runner = kernel_runners[i];
    if (kernel_runner == nullptr) {
      continue;
    }

    if (CheckKernelSupportCapture(kernel_runner, expected_device_context)) {
      if (!find_kernel_can_capture) {
        start = i;
        end = i;
        find_kernel_can_capture = true;
      } else {
        end = i;
      }
    } else {
      if (find_kernel_can_capture) {
        capture_kernel_range_positions_.emplace_back(start, end);
        executors_.emplace_back(CAPTURE_GRAPH, (capture_kernel_range_positions_.size() - static_cast<size_t>(1)));
      }
      executors_.emplace_back(KERNEL, i);
      find_kernel_can_capture = false;
    }
  }

  if (find_kernel_can_capture) {
    capture_kernel_range_positions_.emplace_back(start, end);
    executors_.emplace_back(CAPTURE_GRAPH, (capture_kernel_range_positions_.size() - static_cast<size_t>(1)));
  }

  capture_graph_num_ = capture_kernel_range_positions_.size();
  MS_LOG(INFO) << "Capture graph num: " << capture_graph_num_;
  auto executor_size = executors_.size();
  MS_LOG(DEBUG) << "Dump executor info for capture grpah: ";
  for (size_t i = 0; i < executor_size; i++) {
    std::string executor_mode = (executors_[i].first == CAPTURE_GRAPH ? "capture graph" : "kernel");
    std::ostringstream executor_mode_info;
    if (executors_[i].first == CAPTURE_GRAPH) {
      const auto &range_pair = capture_kernel_range_positions_.at(executors_[i].second);
      executor_mode_info << "executor range:[" << std::to_string(range_pair.first) << ", "
                         << std::to_string(range_pair.second) << "].";
    } else {
      executor_mode_info << "executor order:[" << std::to_string(executors_[i].second) << "]";
    }
    MS_LOG(DEBUG) << "The executor[" << i << "] is " << executor_mode << ", " << executor_mode_info.str();
  }

  return capture_graph_num_ > 0;
}

void GraphCaptureManager::Initialize(const DeviceContext *device_context) {
  if (init_) {
    MS_LOG(EXCEPTION) << "GraphCaptureManager has already initialized.";
  }

  for (size_t i = 0; i < capture_graph_num_; i++) {
    capture_graphs_.push_back(device_context->device_res_manager_->CreateCaptureGraph());
  }
  if (!capture_graphs_.empty()) {
    capture_graph_ = capture_graphs_.front();
    MS_EXCEPTION_IF_NULL(capture_graph_);
  }

  init_ = true;
}

void GraphCaptureManager::Reset(const DeviceContext *device_context) {
  if (capture_graph_ && capture_graph_->HasCapturedGraph()) {
    capture_graphs_.clear();

    for (size_t i = 0; i < capture_graph_num_; i++) {
      capture_graphs_.push_back(device_context->device_res_manager_->CreateCaptureGraph());
    }

    if (!capture_graphs_.empty()) {
      capture_graph_ = capture_graphs_.front();
      MS_EXCEPTION_IF_NULL(capture_graph_);
    }
  }
  if (!fixed_addrs_for_update_.empty()) {
    fixed_addrs_for_update_.clear();
  }
  if (!fixed_addrs_for_set_inputs_.empty()) {
    fixed_addrs_for_set_inputs_.clear();
  }
  if (!weight_kv_addrs_.empty()) {
    weight_kv_addrs_.clear();
  }
}

bool GraphCaptureManager::LaunchAllKernelsWithCapture(OpContext<KernelTensor> *const context,
                                                      const std::vector<KernelRunnerPtr> &kernel_runners,
                                                      SuperKernelActor *super_kernel_actor) {
  MS_LOG(INFO) << "Begin launch all kernels with capture graph.";
  size_t executor_num = executors_.size();
  for (size_t i = 0; i < executor_num; i++) {
    auto &executor = executors_[i];
    if (executor.first == CAPTURE_GRAPH) {
      size_t start = capture_kernel_range_positions_[executor.second].first;
      size_t end = capture_kernel_range_positions_[executor.second].second;
      capture_graphs_[executor.second]->CaptureBegin(0);
      MS_LOG(DEBUG) << "Begin captrue graph, executor index: " << i << ", range[" << start << ", " << end << "].";

      for (size_t j = start; j <= end; j++) {
        const auto &kernel_runner = kernel_runners[j];
        if (kernel_runner == nullptr) {
          continue;
        }

        if (!super_kernel_actor->LaunchKernel(context, kernel_runner, true)) {
          MS_LOG(ERROR) << "Launch kernel in capture mode failed: " << kernel_runner->kernel()->fullname_with_scope();
          return false;
        }
      }
      capture_graphs_[executor.second]->CaptureEnd(0);
      MS_LOG(DEBUG) << "Begin replay captrue graph, executor index: " << i << ", range[" << start << ", " << end
                    << "].";
      capture_graphs_[executor.second]->ExecuteCaptureGraph(0);
    } else {
      auto &kernel_runner = kernel_runners[executor.second];
      MS_LOG(DEBUG) << "Begin launch kernel, executor order index: " << executor.second
                    << ", kernel: " << kernel_runner->kernel()->fullname_with_scope();
      if (!super_kernel_actor->LaunchKernel(context, kernel_runner, true)) {
        MS_LOG(ERROR) << "Launch kernel failed: " << kernel_runner->kernel()->fullname_with_scope();
        return false;
      }
    }
  }
  MS_LOG(INFO) << "End launch all kernels with capture graph.";
  return true;
}

bool GraphCaptureManager::LaunchAllKernelsWithReplayGraph(OpContext<KernelTensor> *const context,
                                                          const std::vector<KernelRunnerPtr> &kernel_runners,
                                                          SuperKernelActor *super_kernel_actor) {
  MS_LOG(INFO) << "Begin launch all kernels with replay graph.";
  size_t executor_num = executors_.size();
  for (size_t i = 0; i < executor_num; i++) {
    auto &executor = executors_[i];
    if (executor.first == CAPTURE_GRAPH) {
      capture_graphs_[executor.second]->ExecuteCaptureGraph(0);
    } else {
      auto &kernel_runner = kernel_runners[executor.second];
      if (!super_kernel_actor->LaunchKernel(context, kernel_runner, true)) {
        MS_LOG(ERROR) << "Launch kernel failed: " << kernel_runner->kernel()->fullname_with_scope();
        return false;
      }
    }
  }
  MS_LOG(INFO) << "End launch all kernels with replay graph.";
  return true;
}

void GraphCaptureManager::HandleFirstUserMemoryFree(const KernelTensorPtr &kernel_tensor,
                                                    const KernelRunnerPtr &kernel_actor,
                                                    std::queue<std::vector<KernelTensorPtr>> *memory_free_lists) {
  if (ActorDispatcher::enable_use_trace_memory() && kernel_tensor->new_ref_count() != SIZE_MAX) {
    memory_free_lists->back().emplace_back(kernel_tensor);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS) << "Add memory free list for tensor:" << kernel_tensor->ToString();
  }
}

bool GraphCaptureManager::IsWeightOrKVCache(GraphParameterStore *cur_graph_parameter_store, const AnfNodePtr &node,
                                            size_t parameter_idx) {
  bool is_weight = cur_graph_parameter_store->GetPositionWeight(parameter_idx);
  std::string cur_node_name = node->fullname_with_scope();
  bool is_kv_cache =
    (cur_node_name.find("key_cache") != std::string::npos || cur_node_name.find("value_cache") != std::string::npos);
  return is_weight || is_kv_cache;
}

void GraphCaptureManager::FetchAllInputsBeforeCaptureGraph(
  OpContext<KernelTensor> *const context, size_t stream_id, const std::vector<KernelRunnerPtr> &kernel_runners,
  std::queue<std::vector<KernelTensorPtr>> *memory_free_lists) {
  MS_LOG(INFO) << "Begin fetch all kernels inputs before capture graph.";
  size_t kernel_num = kernel_runners.size();
  auto cur_graph_parameter_store = ParameterStore::GetInstance().GetGraphParameterStore();
  MS_EXCEPTION_IF_NULL(cur_graph_parameter_store);
  for (size_t i = 0; i < kernel_num; i++) {
    const auto &kernel_actor = kernel_runners[i];
    if (kernel_actor == nullptr) {
      continue;
    }
    for (const auto &parameter_index : kernel_actor->parameter_indexs()) {
      size_t kernel_input_index = parameter_index.first;
      auto outer_index = parameter_index.second.second;
      auto node = parameter_index.second.first.first;
      bool is_first_user = kernel_actor->is_first_used_params()[kernel_input_index];
      auto kernel_tensor =
        FetchParameter(parameter_index.second, context, kernel_actor->GetAID(), is_first_user, stream_id, false);
      const auto &device_tensor = kernel_tensor->device_address();
      MS_EXCEPTION_IF_NULL(device_tensor);
      auto cur_device_context = kernel_actor->device_contexts()[0];
      MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
        << "Actor: " << kernel_actor->GetAID().Name() << ", input index: " << parameter_index.first
        << ", device tensor: " << device_tensor << ", ptr: " << device_tensor->GetPtr()
        << " new ref count:" << device_tensor->new_ref_count()
        << " super kernel actor context:" << cur_device_context->device_context_key().ToString()
        << " kernel actor context:" << cur_device_context->device_context_key().ToString();
      auto real_input_data_infos = kernel_actor->real_input_data_infos();
      auto &real_input_info = real_input_data_infos[kernel_input_index];
      if ((device_tensor->GetDeviceType() != cur_device_context->GetDeviceType()) ||
          !AnfAlgo::IsEquivalentFormat(kernel_tensor->format(), real_input_info->format_) ||
          device_tensor->type_id() != real_input_info->type_id_) {
        MS_EXCEPTION(RuntimeError) << "Does not support heterogeneous scenarios";
      }
      // deal weight/KV Cache
      if (IsWeightOrKVCache(cur_graph_parameter_store, node, outer_index)) {
        // Save the weight or kv value for the subsequent CheckWeightAndKVCacheNotChange function.
        if (weight_kv_addrs_.find(parameter_index.second.first) == weight_kv_addrs_.end()) {
          weight_kv_addrs_[parameter_index.second.first] = {kernel_tensor, parameter_index.second.second, kernel_actor};
        }
        kernel_actor->SetInputDeviceTensor(kernel_tensor, parameter_index.first);
        continue;
      }
      // deal with mormal inputs
      if (fixed_addrs_for_set_inputs_.find(parameter_index.second.first) == fixed_addrs_for_set_inputs_.end()) {
        auto strategy = kernel_actor->get_strategy();
        auto fix_kernel_tensor = AnfAlgo::CreateKernelTensor(
          kernel_tensor->GetShape(), kernel_tensor->GetType(), kernel_tensor->GetValueTrack(), nullptr,
          real_input_info->size_, kernel::GetFormatFromEnumToStr(real_input_info->format_), real_input_info->type_id_,
          real_input_info->shape_, cur_device_context->device_context_key().device_name_,
          cur_device_context->device_context_key().device_id_, device_tensor->user_data());
        MS_EXCEPTION_IF_NULL(kernel_tensor->GetShape());
        fix_kernel_tensor->SetShape(kernel_tensor->GetShape()->Clone());
        fix_kernel_tensor->set_size(device_tensor->GetSize());
        auto fix_device_tensor = fix_kernel_tensor->device_address();
        if (fix_device_tensor->GetPtr() == nullptr) {
          device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, kernel_actor->GetAID().Name(),
                                                         memory::mem_pool::MemType::kOther,
                                                         fix_device_tensor->GetSize(), fix_device_tensor.get());
          if (!cur_device_context->device_res_manager_->AllocateMemory(fix_device_tensor.get(), kDefaultStreamIndex)) {
            SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(strategy, *context, *(cur_device_context),
                                                        kernel_actor->GetAID().Name(), fix_device_tensor->GetSize());
          }
        }
        if (!AsyncCopy(fix_device_tensor.get(), device_tensor.get(), stream_id)) {
          MS_LOG(EXCEPTION) << "Async copy failed, src kernel tensor: " << kernel_tensor->ToString()
                            << ", dst kernel tensor: " << fix_kernel_tensor->ToString();
        }
        // The fixed_addrs_for_set_inputs_ is to set input for kernel actors during the capture phase.
        fixed_addrs_for_set_inputs_[parameter_index.second.first] = fix_kernel_tensor;
        // The fixed_addrs_for_update_ is to update the fix_addr again before the replay phase.
        fixed_addrs_for_update_.emplace_back(parameter_index, fix_kernel_tensor, kernel_actor);
      }
      kernel_actor->SetInputDeviceTensor(fixed_addrs_for_set_inputs_[parameter_index.second.first],
                                         parameter_index.first);

      if (is_first_user) {
        HandleFirstUserMemoryFree(kernel_tensor, kernel_actor, memory_free_lists);
      }
    }
  }
}

void GraphCaptureManager::UpdateFixAddressBeforeReplayGraph(
  OpContext<KernelTensor> *const context, size_t stream_id,
  std::queue<std::vector<KernelTensorPtr>> *memory_free_lists) {
  MS_LOG(INFO) << "Begin update all fixed inputs before replay graph.";
  for (const auto &fix_pair : fixed_addrs_for_update_) {
    auto parameter_index = std::get<kIndex0>(fix_pair);
    auto fix_kernel_tensor = std::get<kIndex1>(fix_pair);
    auto kernel_actor = std::get<kIndex2>(fix_pair);
    size_t kernel_input_index = parameter_index.first;
    MS_EXCEPTION_IF_NULL(kernel_actor);
    MS_EXCEPTION_IF_NULL(fix_kernel_tensor);
    auto cur_device_context = kernel_actor->device_contexts()[0];
    auto real_input_data_infos = kernel_actor->real_input_data_infos();
    auto &real_input_info = real_input_data_infos[kernel_input_index];
    bool is_first_user = kernel_actor->is_first_used_params()[parameter_index.first];
    auto kernel_tensor =
      FetchParameter(parameter_index.second, context, kernel_actor->GetAID(), true, stream_id, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    const auto &device_tensor = kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(device_tensor);
    MS_VLOG(VL_RUNTIME_FRAMEWORK_DEVICE_ADDRESS)
      << "Actor: " << kernel_actor->GetAID().Name() << ", input index: " << parameter_index.first
      << ", device tensor: " << device_tensor << ", ptr: " << device_tensor->GetPtr()
      << " new ref count:" << device_tensor->new_ref_count()
      << " super kernel actor context:" << cur_device_context->device_context_key().ToString();
    if ((device_tensor->GetDeviceType() != cur_device_context->GetDeviceType()) ||
        !AnfAlgo::IsEquivalentFormat(kernel_tensor->format(), real_input_info->format_) ||
        device_tensor->type_id() != real_input_info->type_id_) {
      MS_EXCEPTION(RuntimeError) << "Does not support heterogeneous scenarios";
    }
    if (!AsyncCopy(fix_kernel_tensor->device_address().get(), device_tensor.get(), stream_id)) {
      MS_LOG(EXCEPTION) << "Async copy failed, src kernel tensor: " << kernel_tensor->ToString()
                        << ", dst kernel tensor: " << fix_kernel_tensor->ToString();
    }
    if (is_first_user) {
      HandleFirstUserMemoryFree(kernel_tensor, kernel_actor, memory_free_lists);
    }
  }
}

bool GraphCaptureManager::CheckWeightAndKVCacheNotChange(OpContext<KernelTensor> *const context, size_t stream_id) {
  for (const auto &weight_kv_addr : weight_kv_addrs_) {
    auto old_kernel_tensor = std::get<kIndex0>(weight_kv_addr.second);
    auto outer_idx = std::get<kIndex1>(weight_kv_addr.second);
    auto kernel_actor = std::get<kIndex2>(weight_kv_addr.second);
    auto kernel_tensor =
      FetchParameter({weight_kv_addr.first, outer_idx}, context, kernel_actor->GetAID(), true, stream_id, false);
    if (kernel_tensor->GetSize() != old_kernel_tensor->GetSize() ||
        kernel_tensor->device_ptr() != old_kernel_tensor->device_ptr() ||
        kernel_tensor->GetShape() != old_kernel_tensor->GetShape()) {
      MS_LOG(ERROR) << "KV or Weight device address has changed!!!";
      return false;
    }
  }
  return true;
}

void GraphCaptureManager::Finalize() {
  capture_graph_ = nullptr;
  if (!capture_graphs_.empty()) {
    capture_graphs_.clear();
  }
  if (!fixed_addrs_for_update_.empty()) {
    fixed_addrs_for_update_.clear();
  }
  if (!fixed_addrs_for_set_inputs_.empty()) {
    fixed_addrs_for_set_inputs_.clear();
  }
  if (!weight_kv_addrs_.empty()) {
    weight_kv_addrs_.clear();
  }
}
}  // namespace runtime
}  // namespace mindspore
