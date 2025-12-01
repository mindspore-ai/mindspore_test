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

#include "runtime/core/actors/remote_memory/mem_use_analyzer.h"

#include "runtime/core/actors/remote_memory/mem_counted_cache.h"
#include "runtime/core/actors/remote_memory/mem_action_mgr.h"
#include "runtime/core/actors/control_flow/condition_switch_runner.h"
#include "mindspore/ops/op_def/framework_ops.h"

namespace mindspore {
namespace runtime {

namespace {
void ProcessConditionGatherNode(std::vector<ConditionSwitchInfoPtr> *switch_index,
                                std::vector<KernelRunnerPtr> *true_branch_kernels,
                                std::vector<KernelRunnerPtr> *false_branch_kernels,
                                std::vector<KernelRunnerPtr> *kernel_actors_, const KernelRunnerPtr &kernel_actor) {
  MS_EXCEPTION_IF_NULL(switch_index);
  MS_EXCEPTION_IF_NULL(true_branch_kernels);
  MS_EXCEPTION_IF_NULL(false_branch_kernels);
  MS_EXCEPTION_IF_NULL(kernel_actors_);
  MS_EXCEPTION_IF_NULL(kernel_actor);
  std::vector<KernelRunnerPtr> switch_kernel_actors;
  auto lastest_switch_info = switch_index->back();
  MS_EXCEPTION_IF_NULL(lastest_switch_info);
  auto cur_condition_switch_kernel = lastest_switch_info->kernel_actor_ptr;
  MS_EXCEPTION_IF_NULL(cur_condition_switch_kernel);
  // Mark true branch range
  auto true_branch_expect_node_size = lastest_switch_info->true_branch_node_nums;
  auto true_branch_stored_node_size = true_branch_kernels->size();
  if (true_branch_expect_node_size > true_branch_stored_node_size) {
    MS_LOG(EXCEPTION) << "Invalid true branch node num, expect: " << true_branch_expect_node_size
                      << ", stored: " << true_branch_stored_node_size;
  }
  switch_kernel_actors.insert(switch_kernel_actors.end(), true_branch_kernels->end() - true_branch_expect_node_size,
                              true_branch_kernels->end());
  true_branch_kernels->resize(true_branch_stored_node_size - true_branch_expect_node_size);
  // Mark false branch range
  auto false_branch_expect_node_size = lastest_switch_info->false_branch_node_nums;
  auto false_branch_stored_node_size = false_branch_kernels->size();
  if (false_branch_expect_node_size > false_branch_stored_node_size) {
    MS_LOG(EXCEPTION) << "Invalid false branch node num, expect: " << false_branch_expect_node_size
                      << ", stored: " << false_branch_stored_node_size;
  }
  switch_kernel_actors.insert(switch_kernel_actors.end(), false_branch_kernels->end() - false_branch_expect_node_size,
                              false_branch_kernels->end());
  false_branch_kernels->resize(false_branch_stored_node_size - false_branch_expect_node_size);
  MS_VLOG(VL_REMOTE_MEM_INFO) << "Switch true branch size: " << true_branch_expect_node_size
                              << ", false branch size: " << false_branch_expect_node_size;
  switch_kernel_actors.emplace_back(kernel_actor);
  switch_index->pop_back();

  if (switch_index->empty()) {
    // Single control flow, just append to the kernel_actors_
    kernel_actors_->insert(kernel_actors_->end(), switch_kernel_actors.begin(), switch_kernel_actors.end());
  } else {
    // Nested control flow, choose branch according to the outer switch
    auto outter_switch_info = switch_index->back();
    MS_EXCEPTION_IF_NULL(outter_switch_info);
    bool is_true_branch = cur_condition_switch_kernel->GetEnablePtr() == outter_switch_info->true_branch_enable_ptr;
    auto switch_kernel_actors_size = switch_kernel_actors.size();
    if (is_true_branch) {
      outter_switch_info->true_branch_node_nums += switch_kernel_actors_size;
      true_branch_kernels->insert(true_branch_kernels->end(), switch_kernel_actors.begin(), switch_kernel_actors.end());
    } else {
      outter_switch_info->false_branch_node_nums += switch_kernel_actors_size;
      false_branch_kernels->insert(false_branch_kernels->end(), switch_kernel_actors.begin(),
                                   switch_kernel_actors.end());
    }
  }
}
}  // namespace

KernelTensorPtr MemUseAnalyzer::FindDeviceKernelTensor(const KernelTensorPtr &kernel_tensor) {
  MS_EXCEPTION_IF_NULL(kernel_tensor);
  auto new_kernel_tensor = kernel_tensor;
  auto iter = original_tensors_copyed_map_.find(new_kernel_tensor);
  if (iter != original_tensors_copyed_map_.end()) {
    new_kernel_tensor = iter->second;
  }
  return new_kernel_tensor;
}

std::vector<std::pair<KernelTensorPtr, size_t>> MemUseAnalyzer::GetDeviceKernelTensorsInfo(
  size_t idx, const KernelTensorPtrList &kernel_tensors, bool need_pop_user) {
  std::vector<std::pair<KernelTensorPtr, size_t>> kernel_users_info;
  size_t min_invalid_idx = idx;
  size_t max_invalid_idx = idx;
  if (!latest_switch_infos_.empty()) {
    auto cur_switch_info = latest_switch_infos_.back();
    min_invalid_idx =
      cur_switch_info->is_true_branch ? cur_switch_info->start_false_idx : cur_switch_info->start_true_idx;
    max_invalid_idx = cur_switch_info->is_true_branch ? cur_switch_info->end_false_idx : cur_switch_info->end_true_idx;
    MS_VLOG(VL_REMOTE_MEM_INFO) << "Switch idx: " << idx << " , min_invalid_idx: " << min_invalid_idx
                                << " ,  max_invalid_idx: " << max_invalid_idx;
  }
  for (const auto &kernel_tensor : kernel_tensors) {
    if (kernel_tensor == nullptr) {
      continue;
    }
    bool exists = std::find_if(kernel_users_info.begin(), kernel_users_info.end(),
                               [&kernel_tensor](const std::pair<KernelTensorPtr, size_t> &pair) {
                                 return pair.first == kernel_tensor;
                               }) != kernel_users_info.end();
    if (exists) {
      continue;
    }
    auto iter = kernel_tensor_info_.find(kernel_tensor);
    if (iter == kernel_tensor_info_.end()) {
      // Constant value or param
      continue;
    }
    auto &user_lists = iter->second;
    auto new_kernel_tensor = FindDeviceKernelTensor(kernel_tensor);
    if (user_lists.empty()) {
      MS_VLOG(VL_REMOTE_MEM_INFO) << "Kernel tensor info: " << new_kernel_tensor << ", has no next user";
      kernel_users_info.emplace_back(new_kernel_tensor, SIZE_MAX);
      continue;
    }
    auto next_user_idx = user_lists.front();
    while (next_user_idx <= idx || (next_user_idx <= max_invalid_idx && next_user_idx >= min_invalid_idx)) {
      if (next_user_idx <= max_invalid_idx && next_user_idx >= min_invalid_idx) {
        MS_VLOG(VL_REMOTE_MEM_INFO) << "Skip idx: " << next_user_idx << ", max_invalid_idx: " << max_invalid_idx
                                    << ", min_invalid_idx: " << min_invalid_idx;
      }
      user_lists.pop();
      next_user_idx = user_lists.empty() ? SIZE_MAX : user_lists.front();
    }

    MS_VLOG(VL_REMOTE_MEM_INFO) << "Kernel tensor info: " << new_kernel_tensor << ", next user idx: " << next_user_idx;
    kernel_users_info.emplace_back(new_kernel_tensor, next_user_idx);
    if (need_pop_user && !user_lists.empty()) {
      user_lists.pop();
    }
  }
  return kernel_users_info;
}

std::vector<std::pair<KernelTensorPtr, size_t>> MemUseAnalyzer::GetKernelTensorUserInfo(size_t idx,
                                                                                        KernelRunner *kernel_actor,
                                                                                        bool need_output,
                                                                                        bool need_pop_user) {
  auto kernel_tensors = need_output ? kernel_actor->output_kernel_tensors() : kernel_actor->input_kernel_tensors();
  return GetDeviceKernelTensorsInfo(idx, kernel_tensors, need_pop_user);
}

void MemUseAnalyzer::UpdateCopyKernelTensors(const KernelTensorPtrPairList &kernel_tensors_pair) {
  for (const auto &tensor_pair : kernel_tensors_pair) {
    auto &src_kernel_tensor = tensor_pair.first;
    auto &dst_kernel_tensor = tensor_pair.second;
    // Original kernel tensor vs moved_out kernel tensor
    // ==> original_tensors_copyed_map_[Original_Y] = Host_Y1
    // ==> copyed_tensors_original_map_[Host_Y1] = Original_Y
    if (kernel_tensor_info_.find(src_kernel_tensor) != kernel_tensor_info_.end() &&
        original_tensors_copyed_map_.find(src_kernel_tensor) == original_tensors_copyed_map_.end()) {
      original_tensors_copyed_map_[src_kernel_tensor] = dst_kernel_tensor;
      copyed_tensors_original_map_[dst_kernel_tensor] = src_kernel_tensor;
    } else {
      // Moved out kernel tensor vs move in kernel tensor
      // Map original kernel tensor to move in kernel tensor
      // ==> original_tensors_copyed_map_[Original_Y] = Device_Y2
      // ==> copyed_tensors_original_map_[Device_Y2] = Original_Y
      // ==> erase copyed_tensors_original_map_[Host_Y1]
      auto iter = copyed_tensors_original_map_.find(src_kernel_tensor);
      if (iter == copyed_tensors_original_map_.end()) {
        MS_LOG(EXCEPTION) << "Invailded copy data pair, src: " << src_kernel_tensor << ", dst: " << dst_kernel_tensor;
      }
      auto original_kernel_tensor = iter->second;
      auto original_iter = original_tensors_copyed_map_.find(original_kernel_tensor);
      if (original_iter == original_tensors_copyed_map_.end()) {
        MS_LOG(EXCEPTION) << "Invailded copy data pair, src: " << src_kernel_tensor << ", dst: " << dst_kernel_tensor;
      }
      original_iter->second = dst_kernel_tensor;
      copyed_tensors_original_map_[dst_kernel_tensor] = original_kernel_tensor;
      copyed_tensors_original_map_.erase(src_kernel_tensor);
    }
  }
}

void MemUseAnalyzer::RefreshInputKernelTensors(KernelRunner *kernel_actor) {
  const auto &input_kernel_tensors = kernel_actor->input_kernel_tensors();
  for (size_t index = 0; index < input_kernel_tensors.size(); ++index) {
    auto &kernel_tensor = input_kernel_tensors[index];
    auto new_kernel_tensor = FindDeviceKernelTensor(kernel_tensor);
    if (new_kernel_tensor == kernel_tensor) {
      continue;
    }
    MS_VLOG(VL_REMOTE_MEM_INFO) << "Refresh input kernel tensor: " << kernel_tensor->ToString()
                                << ", new kernel tensor: " << new_kernel_tensor->ToString();
    kernel_tensor->device_address()->SetDevicePtr(new_kernel_tensor->device_address()->GetDevicePtr());
  }
}

void MemUseAnalyzer::ProcessGraphOutputLaunch(const device::DeviceContext *device_context, size_t stream_id) {
  MS_VLOG(VL_REMOTE_MEM_INFO) << "Start processing graph output node.";
  auto current_output_kernel_tensors_info = GetDeviceKernelTensorsInfo(max_idx_, output_kernel_tensors_, true);
  mem_counted_cache_->CheckInputsAvailable(max_idx_, current_output_kernel_tensors_info, device_context);
  (void)mem_counted_cache_->ProcessOutput(device_context, stream_id, max_idx_, {});
  super_kernel_actor_->UpdateOutputKernelTensors(current_output_kernel_tensors_info, output_kernel_tensors_);
  ClearGraphInfo(device_context);
  MS_VLOG(VL_REMOTE_MEM_INFO) << "Finish processing graph output node.";
}

void MemUseAnalyzer::MarkGraphIndex(const std::vector<KernelRunnerPtr> &kernel_actors) {
  std::vector<ConditionSwitchInfoPtr> switch_index;
  std::vector<KernelRunnerPtr> true_branch_kernels;
  std::vector<KernelRunnerPtr> false_branch_kernels;
  kernel_actors_.reserve(kernel_actors.size());
  for (size_t i = 0; i < kernel_actors.size(); ++i) {
    const auto &kernel_actor = kernel_actors[i];
    MS_EXCEPTION_IF_NULL(kernel_actor);
    auto &cnode = kernel_actor->kernel();
    MS_EXCEPTION_IF_NULL(cnode);
    MS_VLOG(VL_REMOTE_MEM_DEBUG) << "Current idx: " << i << ", cnode: " << cnode->DebugString();
    // Not in control flow node
    if (switch_index.empty()) {
      if (!IsPrimitiveCNode(cnode, prim::kPrimConditionSwitch)) {
        kernel_actors_.emplace_back(kernel_actor);
      } else {
        // The first control flow entrance
        auto kernel_actor_ptr = kernel_actor.get();
        MS_EXCEPTION_IF_NULL(kernel_actor_ptr);
        const auto &switch_actor = dynamic_cast<ConditionSwitchRunner *>(kernel_actor_ptr);
        MS_EXCEPTION_IF_NULL(switch_actor);
        auto cur_switch_info =
          std::make_shared<ConditionSwitchInfo>(kernel_actor, &(switch_actor->GetBranchFlag().get()[True]));
        switch_info_map_[kernel_actor_ptr] = cur_switch_info;
        switch_index.push_back(cur_switch_info);
        kernel_actors_.emplace_back(kernel_actor);
      }
      continue;
    }

    // ConditionGather, process the whole control flow
    if (IsPrimitiveCNode(cnode, prim::kPrimConditionGather)) {
      ProcessConditionGatherNode(&switch_index, &true_branch_kernels, &false_branch_kernels, &kernel_actors_,
                                 kernel_actor);
      continue;
    }

    auto &lastest_switch_info = switch_index.back();
    MS_EXCEPTION_IF_NULL(lastest_switch_info);
    if (lastest_switch_info->true_branch_enable_ptr == kernel_actor->GetEnablePtr()) {
      lastest_switch_info->true_branch_node_nums++;
      true_branch_kernels.emplace_back(kernel_actor);
    } else {
      lastest_switch_info->false_branch_node_nums++;
      false_branch_kernels.emplace_back(kernel_actor);
    }

    // Nested control flow scene
    if (IsPrimitiveCNode(cnode, prim::kPrimConditionSwitch)) {
      auto kernel_actor_ptr = kernel_actor.get();
      MS_EXCEPTION_IF_NULL(kernel_actor_ptr);
      const auto &switch_actor = dynamic_cast<ConditionSwitchRunner *>(kernel_actor_ptr);
      MS_EXCEPTION_IF_NULL(switch_actor);
      auto cur_switch_info =
        std::make_shared<ConditionSwitchInfo>(kernel_actor, &(switch_actor->GetBranchFlag().get()[True]));
      switch_info_map_[kernel_actor_ptr] = cur_switch_info;
      switch_index.push_back(cur_switch_info);
    }
  }
}

void MemUseAnalyzer::InitGraphInfo(SuperKernelActor *super_actor, const device::DeviceContext *device_context) {
  const auto &kernel_actors = super_actor->kernel_actors();
  super_kernel_actor_ = super_actor;
  MarkGraphIndex(kernel_actors);
  std::queue<size_t> conditionswitch_idxs;
  // Init not include param and const value, need consider param later
  for (size_t i = 0; i < kernel_actors_.size(); ++i) {
    const auto &kernel_actor = kernel_actors_[i];
    MS_EXCEPTION_IF_NULL(kernel_actor);
    auto kernel_actor_ptr = kernel_actor.get();
    kernel_actor_idx_map_[kernel_actor_ptr] = i;
    auto &cnode = kernel_actor->kernel();
    MS_VLOG(VL_REMOTE_MEM_INFO) << "Process kernel, idx: " << i << ", " << cnode->DebugString();
    if (IsPrimitiveCNode(cnode, prim::kPrimConditionSwitch)) {
      auto &cur_switch_info = switch_info_map_[kernel_actor_ptr];
      cur_switch_info->cur_idx = i;
      cur_switch_info->RefreshNodeIdx();
      conditionswitch_idxs.push(i);
      MS_VLOG(VL_REMOTE_MEM_INFO) << "Cur_switch: " << cur_switch_info->ToString();
    }
    // Process input kernel tensors, add user idx
    for (const auto &real_input : kernel_actor->input_kernel_tensors()) {
      if (real_input == nullptr) {
        continue;
      }
      auto iter = kernel_tensor_info_.find(real_input);
      if (iter == kernel_tensor_info_.end()) {
        MS_VLOG(VL_REMOTE_MEM_INFO) << "Param or const input: " << real_input->ToString();
        continue;
      }
      if (iter->second.empty() || iter->second.back() != i) {
        iter->second.emplace(i);
      }
    }
    // Process output kernel tensors, mark kernel tensor
    for (const auto &real_output : kernel_actor->output_kernel_tensors()) {
      MS_EXCEPTION_IF_NULL(real_output);
      kernel_tensor_info_[real_output] = {};
    }
  }

  // Init copy stream and max mem size
  const auto &kernel_graph = super_kernel_actor_->graph();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  MS_VLOG(VL_REMOTE_MEM_INFO) << "GetDeviceType: " << device_context->GetDeviceType();
  copy_in_stream_id_ = kernel_graph->GetRemoteCopyInStreamId();
  copy_out_stream_id_ = kernel_graph->GetRemoteCopyOutStreamId();
  mem_counted_cache_->SetCopyStreamId(copy_out_stream_id_, copy_in_stream_id_);
  mem_counted_cache_->SetConditionSwitchIdxs(conditionswitch_idxs);
  old_horizon_ = mem_counted_cache_->GetHorizon();
  MS_VLOG(VL_REMOTE_MEM_INFO) << "Finish init graph info: " << kernel_actors.size()
                              << ", copy out stream id: " << copy_out_stream_id_
                              << ", copy in stream id: " << copy_in_stream_id_;

  // Process output kernel tensor
  max_idx_ = kernel_actors.size();
  const auto &graph_output = kernel_graph->output();
  MS_EXCEPTION_IF_NULL(graph_output);
  MS_VLOG(VL_REMOTE_MEM_INFO) << "Output graph node: " << graph_output->DebugString();
  const auto &output_with_indexs = common::AnfAlgo::GetAllOutputWithIndex(graph_output);
  for (const auto &origin_output_with_index : output_with_indexs) {
    const auto &output_with_index = common::AnfAlgo::FetchRealNodeSkipMonadControl(origin_output_with_index);
    const auto &output_node = output_with_index.first;
    MS_EXCEPTION_IF_NULL(output_node);
    if (!output_node->isa<CNode>() || HasAbstractMonad(output_node)) {
      continue;
    }
    auto kernel_tensor = AnfAlgo::GetOutputKernelTensor(output_node, output_with_index.second, false);
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    MS_VLOG(VL_REMOTE_MEM_INFO) << "Output kernel tensor: " << kernel_tensor->ToString();
    auto iter = kernel_tensor_info_.find(kernel_tensor);
    if (iter == kernel_tensor_info_.end()) {
      MS_VLOG(VL_REMOTE_MEM_INFO) << "Invalid output: " << kernel_tensor->ToString();
      (void)output_kernel_tensors_.emplace_back(nullptr);
    } else {
      iter->second.emplace(max_idx_);
      (void)output_kernel_tensors_.emplace_back(kernel_tensor);
    }
  }
}

void MemUseAnalyzer::LaunchTaskBefore(KernelRunner *kernel_actor, const device::DeviceContext *device_context,
                                      bool need_check_output_mem) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  const auto &cnode = kernel_actor->kernel();
  auto idx = kernel_actor_idx_map_[kernel_actor];
  MS_VLOG(VL_REMOTE_MEM_INFO) << "Start launch task before, idx: " << idx << ", cnode: " << cnode->DebugString();
  const auto &kernel_user_info = GetKernelTensorUserInfo(idx, kernel_actor);
  if (!kernel_user_info.empty()) {
    mem_counted_cache_->CheckInputsAvailable(idx, kernel_user_info, device_context);
  }
  const auto &output_kernel_user_info = GetKernelTensorUserInfo(idx, kernel_actor, true, false);
  // Todo: inplace and view op no need
  if (!output_kernel_user_info.empty() && need_check_output_mem) {
    const auto &kernel_tensors_pair =
      mem_counted_cache_->CheckOutputsEnough(idx, output_kernel_user_info, device_context);
    UpdateCopyKernelTensors(kernel_tensors_pair);
  }
  RefreshInputKernelTensors(kernel_actor);
  MS_VLOG(VL_REMOTE_MEM_INFO) << "Finish launch task before, idx: " << idx << ", cnode: " << cnode->DebugString();
}

void MemUseAnalyzer::LaunchTaskAfter(KernelRunner *kernel_actor, const device::DeviceContext *device_context) {
  MS_EXCEPTION_IF_NULL(kernel_actor);
  const auto &cnode = kernel_actor->kernel();
  auto stream_id = kernel_actor->get_stream();
  auto idx = kernel_actor_idx_map_[kernel_actor];
  MS_VLOG(VL_REMOTE_MEM_INFO) << "Start launch task after, idx: " << idx << ", cnode: " << cnode->DebugString();

  // Get condition switch cond result, prepare for the next true/false branch
  if (IsPrimitiveCNode(cnode, prim::kPrimConditionSwitch)) {
    auto &cur_switch_info = switch_info_map_[kernel_actor];
    bool is_true_branch = *kernel_actors_[cur_switch_info->start_true_idx]->GetEnablePtr();
    MS_VLOG(VL_REMOTE_MEM_INFO) << "Kernel cnode: " << cnode->DebugString() << " , " << is_true_branch;
    latest_switch_infos_.emplace_back(cur_switch_info);
    cur_switch_info->is_true_branch = is_true_branch;
    // If execute false branch, set horizon to skip true branch load
    if (!is_true_branch) {
      mem_counted_cache_->SetHorizon(old_horizon_ + cur_switch_info->true_branch_node_nums);
    }
  } else if (IsPrimitiveCNode(cnode, prim::kPrimConditionGather)) {
    latest_switch_infos_.pop_back();
    mem_counted_cache_->SetHorizon(old_horizon_);
  }

  if (!latest_switch_infos_.empty()) {
    auto latest_switch_info = latest_switch_infos_.back();
    auto horizon_refresh_idx =
      std::max(latest_switch_info->start_true_idx, latest_switch_info->end_true_idx - old_horizon_ + 1);
    if (idx == horizon_refresh_idx) {
      mem_counted_cache_->SetHorizon(old_horizon_ + latest_switch_info->false_branch_node_nums);
      MS_VLOG(VL_REMOTE_MEM_INFO) << "Refresh idx: " << horizon_refresh_idx
                                  << " , new: " << old_horizon_ + latest_switch_info->false_branch_node_nums;
    }
  }

  const auto &kernel_users_info = GetKernelTensorUserInfo(idx, kernel_actor, true, true);
  const auto &kernel_tensors_pair =
    mem_counted_cache_->ProcessOutput(device_context, stream_id, idx, kernel_users_info);
  UpdateCopyKernelTensors(kernel_tensors_pair);
  MS_VLOG(VL_REMOTE_MEM_INFO) << "Finish launch task after, idx: " << idx << ", cnode: " << cnode->DebugString();
  // Last launched node, start processing output node, make all output kernel_tensor moved back in device
  if (idx == max_idx_ - 1) {
    ProcessGraphOutputLaunch(device_context, stream_id);
  }
}

void MemUseAnalyzer::ClearGraphInfo(const device::DeviceContext *device_context) {
  MS_VLOG(VL_REMOTE_MEM_INFO) << "Clear graph info";
  kernel_tensor_info_.clear();
  for (auto &[k, v] : original_tensors_copyed_map_) {
    if (k->device_address()->GetDevicePtr() == nullptr) {
      v->device_address()->SetDevicePtr(nullptr);
    }
  }
  original_tensors_copyed_map_.clear();
  copyed_tensors_original_map_.clear();
  kernel_actor_idx_map_.clear();
  output_kernel_tensors_.clear();
  kernel_actors_.clear();
  switch_info_map_.clear();
  latest_switch_infos_.clear();
  mem_counted_cache_->ClearMCCInstance();
}
}  // namespace runtime
}  // namespace mindspore
