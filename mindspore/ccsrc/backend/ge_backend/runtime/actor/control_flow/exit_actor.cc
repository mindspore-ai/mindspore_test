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

#include <algorithm>
#include "backend/ge_backend/runtime/actor/control_flow/exit_actor.h"
#include "backend/ge_backend/runtime/actor/output_actor.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "runtime/device/res_manager/hal_res_manager.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
void ExitActor::Init() {
  // Init output data in base class.
  ControlActor::Init();

  // Init output data in each output branch.
  for (size_t i = 0; i < output_branch_data_arrows_.size(); ++i) {
    auto &output_branch_data_arrows = output_branch_data_arrows_[i];
    for (auto &data_arrow : output_branch_data_arrows) {
      MS_EXCEPTION_IF_NULL(data_arrow);
      auto data = std::make_unique<OpData<KernelTensor>>(data_arrow->to_op_id_, nullptr, data_arrow->to_input_index_);
      (void)output_branch_data_[i].emplace_back(data_arrow->from_output_index_, std::move(data));

      // Identify whether the output data flag is kOutputDataFlagToStack.
      bool is_to_stack = (data_arrow->to_op_id_.Name().find(kStackActorNameSuffix) != std::string::npos);
      size_t output_data_flag = is_to_stack ? kOutputDataFlagToStack : kOutputDataFlagInit;
      (void)output_branch_data_flag_[i].emplace_back(output_data_flag);
    }
  }

  // Check device contexts number.
  if (formal_parameters_.size() != input_kernel_tensors_.size()) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }
}

void ExitActor::FetchInput(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  ControlActor::FetchInput(context);

  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  CopyDeviceAddress(context);
  if (output_branch_dynamic_len_index_.find(output_branch_id_) == output_branch_dynamic_len_index_.end()) {
    auto data_iter = output_branch_data_.find(output_branch_id_);
    if (data_iter != output_branch_data_.end()) {
      for (auto &output_data : data_iter->second) {
        MS_EXCEPTION_IF_NULL(output_data.second);
        if (output_data.first >= input_kernel_tensors_.size()) {
          MS_LOG(EXCEPTION) << "Invalid from index:" << output_data.first << " for actor:" << GetAID()
                            << " to actor:" << output_data.second->op_id_ << " to index:" << output_data.second->index_;
        }
        MS_EXCEPTION_IF_NULL(input_kernel_tensors_[output_data.first]);
        output_data.second->data_ = input_kernel_tensors_[output_data.first];
      }
    }
  } else {
    // The branch id need merge device address.
    MS_LOG(DEBUG) << "Exit actor:" << GetAID() << " merge output";
    MergeDynamiclenDeviceAddress(context);
  }
}

void ExitActor::SendOutput(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  // Before the exit actor sends output, it is necessary to ensure that all reference count calculations in the
  // graph are completed, otherwise the device tensor in the free memory list will be overwritten the next time
  // it is executed, resulting in multiple releases of ptr.
  ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::Wait, context, GetAID());
}

void ExitActor::OnMemoryAllocFinish(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  if (IsRunningFailed(context)) {
    return;
  }

  last_step_created_kernel_tensors_.clear();
  // 1.Send output in base class.
  ControlActor::SendOutput(context);

  // 2.Send output data in output branch.
  const auto &branch_data_iter = output_branch_data_.find(output_branch_id_);
  if (branch_data_iter != output_branch_data_.end()) {
    MS_EXCEPTION_IF_CHECK_FAIL((output_branch_data_flag_.count(output_branch_id_) > 0),
                               "The output branch id is invalid.");
    const auto &output_data_flags = output_branch_data_flag_[output_branch_id_];
    MS_EXCEPTION_IF_CHECK_FAIL((output_data_flags.size() == branch_data_iter->second.size()),
                               "The output data flag size is wrong.");
    for (size_t i = 0; i < branch_data_iter->second.size(); ++i) {
      const auto &output_data = branch_data_iter->second[i];
      MS_EXCEPTION_IF_NULL(output_data.second);
      // Create a new op data for stack actor.
      if (TEST_FLAG(output_data_flags[i], kOutputDataFlagToStack)) {
        auto to_stack_data = std::make_unique<OpData<KernelTensor>>(
          output_data.second->op_id_, output_data.second->data_, output_data.second->index_);
        (void)to_stack_data_.emplace_back(std::move(to_stack_data));
        ActorDispatcher::Send(output_data.second->op_id_, &OpRTActor::RunOpData, to_stack_data_.back().get(), context);
      } else {
        ActorDispatcher::Send(output_data.second->op_id_, &OpRTActor::RunOpData, output_data.second.get(), context);
      }
    }
  }

  // 3.Send output control in output branch.
  const auto &control_iter = output_branch_control_arrows_.find(output_branch_id_);
  if (control_iter != output_branch_control_arrows_.end()) {
    auto source_aid = const_cast<AID *>(&GetAID());
    for (const auto &control_arrow : control_iter->second) {
      ActorDispatcher::Send(control_arrow, &OpRTActor::RunOpControl, source_aid, context);
    }
  }

  // 3.Send output partial in output branch.
  const auto &partial_iter = output_branch_partial_arrows_.find(output_branch_id_);
  if (partial_iter != output_branch_partial_arrows_.end()) {
    for (const auto &arrow : partial_iter->second) {
      MS_EXCEPTION_IF_NULL(arrow);
      if (IntToSize(arrow->from_output_index_) >= input_partials_.size()) {
        std::string error_info = "Invalid partial input:" + std::to_string(arrow->from_output_index_) +
                                 " current:" + std::to_string(input_partials_.size()) + " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      auto output_partial = input_partials_[IntToSize(arrow->from_output_index_)];
      ActorDispatcher::Send(arrow->to_op_id_, &ControlActor::RunOpPartial, output_partial,
                            IntToSize(arrow->to_input_index_), context);
    }
  }
}

void ExitActor::IncreaseDynamicRefCounts(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  ControlActor::IncreaseDynamicRefCounts(context);

  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  // Increase dynamic ref count by the output data in output branch.
  if (output_branch_data_.count(output_branch_id_) > 0) {
    for (auto &output_data : output_branch_data_[output_branch_id_]) {
      MS_EXCEPTION_IF_NULL(output_data.second);
      IncreaseDynamicRefCount(output_data.second.get());
    }
  }

  // Increase dynamic ref count by the output partial in output branch.
  if (output_branch_partial_arrows_.count(output_branch_id_) > 0) {
    for (const auto &partial_arrow : output_branch_partial_arrows_[output_branch_id_]) {
      MS_EXCEPTION_IF_NULL(partial_arrow);
      if (IntToSize(partial_arrow->from_output_index_) >= input_partials_.size()) {
        std::string error_info = "Invalid partial input:" + std::to_string(partial_arrow->from_output_index_) +
                                 " current:" + std::to_string(input_partials_.size()) + " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      auto output_partial = input_partials_[IntToSize(partial_arrow->from_output_index_)];
      IncreaseDynamicRefCount(output_partial);
    }
  }
  if (input_kernel_tensors_.size() != formal_parameters_.size()) {
    MS_LOG(ERROR) << "Input device tensor size:" << input_kernel_tensors_.size()
                  << " is not equal to parameter size:" << formal_parameters_.size() << " for actor:" << GetAID();
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  auto device_type = device::GetDeviceTypeByName(ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET));
  device::ResKey res_key{device_type, device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);

  // The input device tensor may not have users and needs to free the memory.
  for (size_t i = 0; i < input_kernel_tensors_.size(); ++i) {
    if ((input_kernel_tensors_[i] != nullptr) && (input_kernel_tensors_[i]->device_address() != nullptr) &&
        (device_type != input_kernel_tensors_[i]->GetDeviceType())) {
      MS_LOG(EXCEPTION) << "GE backend only support Ascend, but get " << input_kernel_tensors_[i]->device_name();
    }
    if ((input_kernel_tensors_[i] != nullptr) && (input_kernel_tensors_[i]->device_address() != nullptr) &&
        (input_kernel_tensors_[i]->device_address()->dynamic_ref_count() == 0)) {
      MS_LOG(INFO) << GetAID().Name() << " input index:" << i << " has no user and free the memory.";
      res_manager->FreeMemory(input_kernel_tensors_[i]->device_address().get());
    }
  }
}

void ExitActor::MergeDynamiclenDeviceAddress(OpContext<KernelTensor> *const context) {
  if (output_branch_dynamic_len_index_.find(output_branch_id_) == output_branch_dynamic_len_index_.end()) {
    return;
  }
  auto real_indexes = output_branch_dynamic_len_index_[output_branch_id_];
  std::vector<OpPartialPtr> new_partials;
  std::vector<KernelTensorPtr> new_kernel_tensors;
  // Collect the new output of actor, merge the device address for dynamic len.
  for (size_t i = 0; i < real_indexes.size(); ++i) {
    const auto &indexes = real_indexes[i].first;
    if (real_indexes[i].second) {
      std::vector<KernelTensor *> addr_list;
      for (size_t index : indexes) {
        if (index >= input_kernel_tensors_.size()) {
          std::string error_info = "Invalid real index:" + std::to_string(index) + " for index:" + std::to_string(i) +
                                   " total size:" + std::to_string(input_kernel_tensors_.size()) +
                                   " for actor:" + GetAID().Name();
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
        }
        if (input_kernel_tensors_[index] == nullptr || input_kernel_tensors_[index]->device_address() == nullptr) {
          std::string error_info =
            "Invalid input device address index:" + std::to_string(index) + " for index:" + std::to_string(i) +
            " total size:" + std::to_string(input_kernel_tensors_.size()) + " for actor:" + GetAID().Name();
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
        }
        addr_list.emplace_back(input_kernel_tensors_[index].get());
      }
      KernelTensorPtr new_kernel_tensor = nullptr;
      MergeDeviceAddress(context, addr_list, &new_kernel_tensor);
      MS_EXCEPTION_IF_NULL(new_kernel_tensor);
      new_kernel_tensors.emplace_back(new_kernel_tensor);
      new_partials.emplace_back(nullptr);
    } else if (indexes.empty() || indexes[0] >= input_partials_.size()) {
      std::string error_info = "Invalid index num:" + std::to_string(indexes.size()) +
                               " for index:" + std::to_string(i) + " for actor:" + GetAID().Name();
      MS_LOG(WARNING) << error_info;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    } else if (input_partials_[indexes[0]] != nullptr) {
      new_kernel_tensors.emplace_back(nullptr);
      new_partials.emplace_back(input_partials_[indexes[0]]);
    } else if (input_kernel_tensors_[indexes[0]] != nullptr &&
               input_kernel_tensors_[indexes[0]]->device_address() != nullptr) {
      new_kernel_tensors.emplace_back(input_kernel_tensors_[indexes[0]]);
      new_partials.emplace_back(nullptr);
    } else {
      std::string error_info = "Failed to get input for real index:" + std::to_string(indexes[0]) +
                               " for index:" + std::to_string(i) + " for actor:" + GetAID().Name();
      MS_LOG(WARNING) << error_info;
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }
  auto data_iter = output_branch_data_.find(output_branch_id_);
  if (data_iter != output_branch_data_.end()) {
    for (auto &output_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(output_data.second);
      if (output_data.first >= new_kernel_tensors.size()) {
        MS_EXCEPTION_IF_NULL(output_data.second);
        MS_LOG(EXCEPTION) << "Invalid from index:" << output_data.first << " for actor:" << GetAID()
                          << " to actor:" << output_data.second->op_id_ << " to index:" << output_data.second->index_;
      }
      MS_EXCEPTION_IF_NULL(new_kernel_tensors[output_data.first]);
      output_data.second->data_ = new_kernel_tensors[output_data.first];
    }
  }
  for (size_t i = 0; i < new_partials.size() && i < input_partials_.size(); ++i) {
    if (new_partials[i] != nullptr) {
      MS_LOG(DEBUG) << "Set op partial for index:" << i << " actor:" << GetAID();
      input_partials_[i] = new_partials[i];
    }
  }
}

bool ExitActor::IsNeedCopyDeviceAddress(DeviceTensor *const input_device_tensor, size_t index) {
  if ((input_device_tensor == nullptr) || (!is_need_copy_device_tensors_[index])) {
    return false;
  }

  if (is_need_dynamic_checks_[index]) {
    if (input_device_tensor->dynamic_ref_count() != INT32_MAX) {
      return false;
    }
    const auto &node = input_device_tensor->GetNodeIndex().first;
    if (node != nullptr) {
      if (!node->isa<CNode>()) {
        MS_LOG(DEBUG) << "Input device address:" << input_device_tensor << " ptr:" << input_device_tensor->GetPtr()
                      << " for node:" << node->DebugString() << " is not need replace ptr for actor:" << GetAID();
        return false;
      }
      const auto &iter = ref_out_in_map_.find(input_device_tensor->GetNodeIndex());
      if (iter != ref_out_in_map_.end() && iter->second.first != nullptr && (!iter->second.first->isa<CNode>())) {
        MS_LOG(DEBUG) << "Input device address:" << input_device_tensor << " ptr:" << input_device_tensor->GetPtr()
                      << " for node:" << node->DebugString()
                      << " is a ref node of:" << iter->second.first->DebugString()
                      << " not need replace ptr for actor:" << GetAID();
        return false;
      }
    }
  }
  return true;
}

void ExitActor::UpdateDeviceOutputData() {
  for (size_t i = 0; i < output_data_by_output_index_.size(); ++i) {
    if (output_data_by_output_index_[i].empty()) {
      continue;
    }

    const auto &kernel_tensor = input_kernel_tensors_[i];
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    for (auto &output_data : output_data_by_output_index_[i]) {
      MS_EXCEPTION_IF_NULL(output_data);
      output_data->data_ = kernel_tensor;
    }
  }
}

void ExitActor::CopyDeviceAddress(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  // If node is not empty, it is the exit of funcgraph, no need to create device address.
  if (node_ != nullptr) {
    return;
  }
  if (input_kernel_tensors_.size() != is_need_copy_device_tensors_.size() ||
      input_kernel_tensors_.size() != is_dynamic_shapes_.size() ||
      input_kernel_tensors_.size() != formal_parameters_.size() ||
      input_kernel_tensors_.size() != is_need_dynamic_checks_.size()) {
    std::string error_info = "Invalid input device tensor size:" + std::to_string(input_kernel_tensors_.size()) +
                             " need tensor size:" + std::to_string(is_need_copy_device_tensors_.size()) +
                             " need dynamic shape size:" + std::to_string(is_dynamic_shapes_.size()) +
                             " need context size:" + std::to_string(formal_parameters_.size()) +
                             " for actor:" + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
  }

  auto repeat_output_index = GetRepeatDeviceAddressIndexPair(input_kernel_tensors_);
  std::vector<KernelTensorPtr> new_kernel_tensors;
  for (size_t i = 0; i < input_kernel_tensors_.size(); ++i) {
    auto input_device_tensor =
      input_kernel_tensors_[i] == nullptr ? nullptr : input_kernel_tensors_[i]->device_address().get();
    if (!IsNeedCopyDeviceAddress(input_device_tensor, i)) {
      (void)new_kernel_tensors.emplace_back(input_kernel_tensors_[i]);
      continue;
    }

    auto iter = repeat_output_index.find(i);
    if (iter != repeat_output_index.end()) {
      if (iter->second >= new_kernel_tensors.size()) {
        MS_LOG(EXCEPTION) << "Invalid output index:" << i << " real index:" << iter->second
                          << " for actor:" << GetAID();
      }
      (void)new_kernel_tensors.emplace_back(new_kernel_tensors[iter->second]);
      continue;
    }

    // Update the real used device context by the input data.
    auto ms_context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(ms_context);
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    auto device_type = device::GetDeviceTypeByName(ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET));
    if (device_type != input_device_tensor->GetDeviceType()) {
      device_type = input_device_tensor->GetDeviceType();
    }
    device::ResKey res_key{device_type, device_id};
    auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
    MS_EXCEPTION_IF_NULL(res_manager);

    const KernelWithIndex &node_with_index = input_device_tensor->GetNodeIndex();
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    // Create the new device tensor to take over the input_device_tensors which are the outputs of kernel graphs.
    const auto &kernel_tensor = input_kernel_tensors_[i];
    auto new_kernel_tensor = kernel_tensor->CloneKernelTensor();
    MS_EXCEPTION_IF_NULL(new_kernel_tensor);
    new_kernel_tensor->set_device_ptr(nullptr);
    DeviceTensorPtr new_device_tensor = new_kernel_tensor->device_address();
    MS_EXCEPTION_IF_NULL(new_device_tensor);
    MS_LOG(DEBUG) << "Actor:" << GetAID() << " create new device tensor:" << new_device_tensor
                  << " type:" << new_device_tensor->type_id() << " by input device tensor:" << input_device_tensor
                  << " shape:"
                  << (kernel_tensor->GetShape() == nullptr ? "null" : kernel_tensor->GetShape()->ToString())
                  << (kernel_tensor->GetType() == nullptr ? "null" : kernel_tensor->GetType()->ToString());
    (void)created_kernel_tensors_.emplace_back(new_kernel_tensor);
    (void)new_kernel_tensors.emplace_back(new_kernel_tensor);
    new_device_tensor->set_need_sync_user_data(input_device_tensor->need_sync_user_data());
    new_device_tensor->SetNodeIndex(node_with_index.first, node_with_index.second);
    new_device_tensor->set_from_persistent_mem(input_device_tensor->from_persistent_mem());
    // The device address which is created by actor uses the dynamic ref count.
    new_device_tensor->set_dynamic_ref_count(0);
    new_device_tensor->set_original_ref_count(SIZE_MAX);
    new_device_tensor->ResetRefCount();

    // If the address ptr can't be changed, then alloc the new device memory and copy the data.
    if (input_device_tensor->is_ptr_persisted()) {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), "CopyDeviceAddress", "");
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, GetAID().Name(), memory::mem_pool::MemType::kOther,
                                                     new_device_tensor->GetSize(), new_device_tensor.get());
      if (!res_manager->AllocateMemory(new_device_tensor.get(), kDefaultStreamIndex)) {
        SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *context, GetAID().Name(),
                                                    new_device_tensor->GetSize());
      }
      if (!new_device_tensor->SyncDeviceToDevice(input_device_tensor)) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Sync device to device failed.");
      }
    } else {
      // Move the device ptr from input_device_tensor to new_device_tensor.
      input_device_tensor->Swap(new_device_tensor.get());
      if (new_device_tensor->deleter() == nullptr) {
        new_device_tensor->set_from_mem_pool(true);
      }
    }
    MS_LOG(DEBUG) << GetAID().Name() << " creates the dynamic ref device address:" << new_device_tensor.get()
                  << ", ptr:" << new_device_tensor->GetPtr()
                  << ", from node:" << node_with_index.first->fullname_with_scope()
                  << " with index:" << node_with_index.second;
  }
  input_kernel_tensors_.swap(new_kernel_tensors);
  UpdateDeviceOutputData();
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
