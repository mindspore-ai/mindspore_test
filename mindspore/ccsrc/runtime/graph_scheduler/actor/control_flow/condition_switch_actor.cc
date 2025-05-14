/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/control_flow/condition_switch_actor.h"
#include "runtime/graph_scheduler/actor/control_flow/condition_gather_actor.h"

namespace mindspore {
namespace runtime {
ConditionSwitchActor::ConditionSwitchActor(const std::string &name, const CNodePtr &kernel,
                                           const DeviceContext *device_context, const AID &memory_manager_aid,
                                           const AID *debug_aid, const AID *recorder_aid,
                                           GraphExecutionStrategy strategy,
                                           const std::set<size_t> &modifiable_ref_input_indexes,
                                           const std::set<size_t> &modifiable_ref_output_indexes,
                                           const KernelTransformType &type)
    : KernelActor(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                  modifiable_ref_input_indexes, modifiable_ref_output_indexes, type) {
  need_wait_pipeline_ = true;
}

void ConditionSwitchActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);

  kernel_info_ = dynamic_cast<KernelInfo *>(kernel_->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info_);
  kernel_mod_ = kernel_info_->MutableKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  const auto &output_kernel_tensors = kernel_info_->output_kernel_tensor_list();
  for (size_t i = 0; i < output_kernel_tensors.size(); ++i) {
    auto &output_kernel_tensor = output_kernel_tensors[i];
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    auto output_address = output_kernel_tensor->device_address().get();

    if (output_address->stream_id() != kernel_info_->stream_id()) {
      MS_LOG(DEBUG) << "Output address : " << output_address << " stream id :" << output_address->stream_id()
                    << " is not equal kernel info stream id : " << kernel_info_->stream_id() << ".";
    }
    (void)output_kernel_tensors_.emplace_back(output_kernel_tensor);
  }

  real_input_num_ = common::AnfAlgo::GetInputTensorNum(kernel_);
  InitIsMonadInput();
  InitInputInfo();

  for (size_t index : output_free_index_) {
    if (index >= output_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << index << " output size:" << output_kernel_tensors_.size()
                        << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(output_kernel_tensors_[index]);
  }

  if (!kernel_->HasAttr(kInlineSubGraphName)) {
    MS_LOG(EXCEPTION) << "Failed to get inline graph name by actor:" << GetAID();
  }
  const auto &inline_sub_graph_names = kernel_->GetAttr(kInlineSubGraphName);
  MS_EXCEPTION_IF_NULL(inline_sub_graph_names);
  MS_LOG(DEBUG) << "inline sub graph name:" << inline_sub_graph_names->ToString() << " for actor:" << GetAID();
  if (!inline_sub_graph_names->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Invalid input subgraph name:" << inline_sub_graph_names->ToString()
                      << " for actor:" << GetAID();
  }
  const auto &tuple_name = inline_sub_graph_names->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_name);
  for_each(tuple_name->value().begin(), tuple_name->value().end(),
           [this](const auto &value) { branch_names_.emplace_back(GetValue<std::string>(value)); });
  MS_LOG(DEBUG) << "Branch names:" << branch_names_ << " for actor:" << GetAID();
}

void ConditionSwitchActor::UpdateRefDeviceAddress(OpContext<KernelTensor> *const context, bool increase_ref_count) {
  if (input_kernel_tensors_.size() != output_kernel_tensors_.size() + 1) {
    MS_LOG(EXCEPTION) << "Invalid input tensor size:" << input_kernel_tensors_.size()
                      << " and output device tensor size:" << output_kernel_tensors_.size()
                      << " for actor:" << GetAID();
  }
  for (size_t i = 0; i < output_kernel_tensors_.size(); ++i) {
    if (input_kernel_tensors_[i + 1] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid input device tensor index:" << i + 1 << " for actor:" << GetAID();
    }
    if (output_kernel_tensors_[i] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid input device tensor index:" << i + 1 << " for actor:" << GetAID();
    }
    auto input_device_tensor = input_kernel_tensors_[i + 1]->device_address().get();
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    auto output_device_tensor = output_kernel_tensors_[i]->device_address().get();
    MS_EXCEPTION_IF_NULL(output_device_tensor);
    output_device_tensor->set_pointer_ref_count(input_device_tensor->pointer_ref_count());
    if (input_kernel_tensors_[i + 1]->heterogeneous_info() != nullptr) {
      output_kernel_tensors_[i]->set_heterogeneous_info(std::make_shared<HeterogeneousInfo>());
      *(output_kernel_tensors_[i]->heterogeneous_info()) = *(input_kernel_tensors_[i + 1]->heterogeneous_info());
    }
    output_device_tensor->IncreaseNewRefCount(GetAID().Name());
    MS_LOG(DEBUG) << "Actor:" << GetAID() << " increase new ref count:" << output_device_tensor->new_ref_count()
                  << " and set ref device address:" << output_device_tensor->PrintInfo()
                  << " ref input device address:" << input_device_tensor->PrintInfo();
  }
}

void ConditionSwitchActor::ExecuteInferShapeTask(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(kernel_);
  MS_LOG(DEBUG) << "Begin InferShape for kernel: " << kernel_->fullname_with_scope();
  Async(kernel_async_resize_aid_, &KernelAsyncResizeActor::ResizeKernelMod, context, this);
  MS_LOG(DEBUG) << "End InferShape for kernel: " << kernel_->fullname_with_scope();
}

void ConditionSwitchActor::ExecuteResizeKernelModTask(OpContext<KernelTensor> *const context) {
  Async(kernel_async_launch_aid_, &KernelAsyncLaunchActor::LaunchKernel, context, this);
}

void ConditionSwitchActor::ExecuteLaunchKernelTask(OpContext<KernelTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kKernel, ProfilerEvent::kKernelLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(kernel_);
  MS_LOG(DEBUG) << "Begin launch kernel: " << kernel_->fullname_with_scope();
  if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
    MS_LOG(INFO) << "Run failed and early stop.";
    return;
  }
  MS_EXCEPTION_IF_NULL(input_kernel_tensors_[0]);
  bool index = input_kernel_tensors_[0]->GetValueWithCheck<bool>();
  if (common::IsDryRun()) {
    index = true;
  }
  MS_LOG(DEBUG) << "Index:" << index << " for actor:" << GetAID();
  if (index >= branch_names_.size()) {
    std::string error_info = "Invalid index:" + std::to_string(index) +
                             " and branch size:" + std::to_string(branch_names_.size()) +
                             " for actor:" + GetAID().Name();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), error_info);
  }
  MS_EXCEPTION_IF_NULL(gather_branch_name_);
  *gather_branch_name_ = branch_names_[index];
  MS_EXCEPTION_IF_NULL(branch_flags_);
  branch_flags_.get()[index] = true;
  MS_LOG(DEBUG) << "Enable flag:" << &(branch_flags_.get()[index]) << " by index:" << index
                << " branch name:" << branch_names_[index] << " in actor:" << GetAID();
  new_memory_free_list_.clear();
  for (size_t input_index : input_free_index_) {
    if (input_index >= input_kernel_tensors_.size() || input_kernel_tensors_[input_index] == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get input device tensor index:" << input_index
                        << " total input size:" << input_kernel_tensors_.size()
                        << " for node:" << kernel_->DebugString() << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(input_kernel_tensors_[input_index]);
    MS_LOG(DEBUG) << "Add decrease new ref count for kernel tensor:" << input_kernel_tensors_[input_index]
                  << " in actor:" << GetAID();
  }
  if (branch_output_free_index_.find(branch_names_[index]) != branch_output_free_index_.end()) {
    for (size_t output_index : branch_output_free_index_[branch_names_[index]]) {
      if (output_index >= output_kernel_tensors_.size() || output_kernel_tensors_[output_index] == nullptr) {
        MS_LOG(EXCEPTION) << "Invalid output device tensor index:" << output_index
                          << "total size:" << output_kernel_tensors_.size() << " for actor:" << GetAID();
      }
      new_memory_free_list_.emplace_back(output_kernel_tensors_[output_index]);
      MS_LOG(DEBUG) << "Add decrease new ref count for kernel tensor:" << output_kernel_tensors_[output_index]
                    << " in actor:" << GetAID();
    }
  }

  if (new_memory_free_list_.size() > 0) {
    SendMemoryFreeReq(context);
  }
  MS_LOG(DEBUG) << "End launch kernel: " << kernel_->fullname_with_scope();
}

void ConditionSwitchActor::SendOutput(OpContext<KernelTensor> *const context, size_t index) {
  MS_EXCEPTION_IF_NULL(gather_aid_);
  MS_LOG(DEBUG) << "condition actor run for index:" << index << " branch name:" << branch_names_[index]
                << " for actor:" << GetAID();
  ActorDispatcher::Send(*gather_aid_, &ConditionGatherActor::RunBranchName, branch_names_[index], context);

  if (output_data_arrows_.size() != output_data_nodes_.size() || output_data_nodes_.size() != output_data_.size() ||
      output_data_.size() != output_data_branch_indexes_.size()) {
    MS_LOG(EXCEPTION) << "Invalid data arrow size:" << output_data_arrows_.size()
                      << " node size:" << output_data_nodes_.size() << " data size:" << output_data_.size()
                      << " index size:" << output_data_branch_indexes_.size() << " for actor:" << GetAID();
  }
  for (size_t i = 0; i < output_data_branch_indexes_.size(); ++i) {
    if (TEST_FLAG(output_data_[i].second, kOutputDataFlagToFusion)) {
      if (data_arrow_to_fusion_actor_indexs_.find(output_data_arrows_[i].get()) ==
          data_arrow_to_fusion_actor_indexs_.end()) {
        MS_LOG(EXCEPTION) << "Failed to get real from index by output data arrow from index:"
                          << output_data_arrows_[i]->from_output_index_ << " to " << output_data_arrows_[i]->to_op_id_
                          << " by actor:" << GetAID();
      }
      output_data_[i].first->index_ = SizeToInt(data_arrow_to_fusion_actor_indexs_.at(output_data_arrows_[i].get()));
    }
    if (output_data_branch_indexes_[i] == index) {
      ActorDispatcher::Send(output_data_arrows_[i]->to_op_id_, &OpRTActor::RunOpData, output_data_[i].first.get(),
                            context);
    }
  }

  if (output_control_arrows_.size() != output_control_branch_indexes_.size()) {
    MS_LOG(EXCEPTION) << "Invalid control arrow size:" << output_control_arrows_.size()
                      << output_control_branch_indexes_.size() << " for actor:" << GetAID();
  }
  for (size_t i = 0; i < output_control_branch_indexes_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_control_arrows_[i]);
    if (output_control_branch_indexes_[i] == index) {
      ActorDispatcher::Send(output_control_arrows_[i]->to_op_id_, &OpRTActor::RunOpControl,
                            const_cast<AID *>(&GetAID()), context);
    }
  }
}

void ConditionSwitchActor::Run(OpContext<KernelTensor> *const context) {
  try {
    if (!WaitRuntimePipelineFinish(context, GetAID().Name())) {
      MS_LOG(INFO) << "Run failed and early stop.";
      return;
    }
    MS_LOG(INFO) << "Sync stream in the condition switch.";
    ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
    FetchInput(context);
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[0]);
    MS_EXCEPTION_IF_NULL(input_kernel_tensors_[0]->device_address());
    bool index = input_kernel_tensors_[0]->GetValueWithCheck<bool>();
    if (common::IsDryRun()) {
      index = true;
    }
    MS_LOG(DEBUG) << "Index:" << index << " for actor:" << GetAID();
    if (index >= branch_names_.size()) {
      std::string error_info = "Invalid index:" + std::to_string(index) +
                               " and branch size:" + std::to_string(branch_names_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), error_info);
    }
    EraseInput(context);
    CollectMemoryFreeList(index);
    if (memory_free_list_.size() > 0) {
      SendMemoryFreeReq(context);
    }
    MS_LOG(DEBUG) << "Launch kernel:" << kernel_->fullname_with_scope() << " by index:" << index;
    SendOutput(context, index);
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info =
      "#umsg#Kernel error:#umsg#run kernel[" + kernel_->fullname_with_scope() + "] failed, exception: " + e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), error_info);
  }
}

void ConditionSwitchActor::CollectMemoryFreeList(size_t index) {
  memory_free_list_.clear();
  memory_free_list_.insert(memory_free_list_.end(), input_kernel_tensors_.begin(), input_kernel_tensors_.end());
  memory_free_list_.insert(memory_free_list_.end(), input_kernel_tensors_.begin() + 1, input_kernel_tensors_.end());
  for (size_t i = 0; i < branch_origin_ref_count_.size(); ++i) {
    if (i == index) {
      continue;
    }
    if (branch_origin_ref_count_[i].size() + 1 != input_kernel_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid origin ref count size:" << branch_origin_ref_count_[i]
                        << " and input size:" << input_kernel_tensors_.size() << " for actor:" << GetAID();
    }
    MS_LOG(DEBUG) << "Free memory for branch:" << i << " for actor:" << GetAID();
    for (size_t j = 0; j < branch_origin_ref_count_[i].size(); ++j) {
      std::fill_n(back_inserter(memory_free_list_), branch_origin_ref_count_[i][j], input_kernel_tensors_[j + 1]);
    }
  }
}

void ConditionSwitchActor::FetchParameterInput(OpContext<KernelTensor> *const context) {
  // Fetch parameter input tensor from graph parameter store.
  if (!enable_input_optimize_) {
    return;
  }

  for (auto &parameter_index : parameter_indexs_) {
    auto kernel_tensor = FetchParameter(parameter_index.second, context, GetAID());
    if (kernel_tensor == nullptr) {
      std::string error_info =
        GetAID().Name() + " get graph parameter store failed: " + parameter_index.second.first.first->DebugString() +
        ", device type:" + std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    if (parameter_index.first >= input_kernel_tensors_.size()) {
      std::string error_info = "The input index is out of range, need:" + std::to_string(parameter_index.first) +
                               " current:" + std::to_string(input_kernel_tensors_.size()) +
                               " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_kernel_tensors_[parameter_index.first] = kernel_tensor;
  }
}

void ConditionSwitchActor::FetchInput(OpContext<KernelTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);

  // Fetch input device tensor from input data.
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      if (IntToSize(input_data->index_) >= input_kernel_tensors_.size()) {
        std::string error_info = "Invalid input index, need:" + std::to_string(input_data->index_) +
                                 " current:" + std::to_string(input_kernel_tensors_.size()) +
                                 " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      MS_EXCEPTION_IF_NULL(input_data->data_);
      input_kernel_tensors_[IntToSize(input_data->index_)] = input_data->data_;
    }
  }

  // Fetch input device tensor from device tensor store.
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
    auto kernel_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                device_contexts_[0]->GetDeviceType());
    if (kernel_tensor == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key.second->DebugString() +
        ", device type:" + std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    if (device_tensor_store_key.first >= input_kernel_tensors_.size()) {
      std::string error_info =
        "The input index is out of range, need:" + std::to_string(device_tensor_store_key.first) +
        " current:" + std::to_string(input_kernel_tensors_.size()) + " for actor:" + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    MS_EXCEPTION_IF_NULL(kernel_tensor);
    input_kernel_tensors_[device_tensor_store_key.first] = kernel_tensor;
  }

  FetchParameterInput(context);

  if (output_data_by_output_index_.size() + 1 != input_kernel_tensors_.size()) {
    MS_LOG(EXCEPTION) << "Invalid output size:" << output_data_by_output_index_.size()
                      << " and input device tensor size:" << input_kernel_tensors_.size() << " for actor:" << GetAID();
  }

  for (size_t i = 0; i < output_data_by_output_index_.size(); ++i) {
    if (output_data_by_output_index_[i].empty()) {
      continue;
    }
    const auto &data = input_kernel_tensors_[i + 1];
    MS_EXCEPTION_IF_NULL(data);
    for (auto &output_data : output_data_by_output_index_[i]) {
      MS_EXCEPTION_IF_NULL(output_data);
      output_data->data_ = data;
    }
  }
}
}  // namespace runtime
}  // namespace mindspore
