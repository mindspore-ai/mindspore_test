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

#include "runtime/graph_scheduler/actor/control_flow/condition_gather_actor.h"
#include "include/backend/mem_reuse/mem_tracker.h"

namespace mindspore {
namespace runtime {
ConditionGatherActor::ConditionGatherActor(const std::string &name, const CNodePtr &kernel,
                                           const DeviceContext *device_context, const AID &memory_manager_aid,
                                           const AID *debug_aid, const AID *recorder_aid,
                                           GraphExecutionStrategy strategy,
                                           const std::set<size_t> &modifiable_ref_input_indexes,
                                           const std::set<size_t> &modifiable_ref_output_indexes,
                                           const KernelTransformType &type)
    : KernelActor(name, kernel, device_context, memory_manager_aid, debug_aid, recorder_aid, strategy,
                  modifiable_ref_input_indexes, modifiable_ref_output_indexes, type) {}

ConditionGatherActor::~ConditionGatherActor() {
  for_each(need_clean_ptr_device_addresses_.begin(), need_clean_ptr_device_addresses_.end(),
           [](const device::DeviceAddressPtr &device_address) { device_address->set_ptr(nullptr); });
}

void ConditionGatherActor::RunBranchName(const std::string &branch_name, OpContext<DeviceTensor> *const context) {
  MS_LOG(DEBUG) << "Condition gather actor:" << GetAID() << " branch name:" << branch_name;
  current_branch_name_ = branch_name;
  if (branch_name_to_input_data_num_.find(current_branch_name_) == branch_name_to_input_data_num_.end()) {
    input_datas_num_ = 0;
  } else {
    input_datas_num_ = branch_name_to_input_data_num_[current_branch_name_];
  }
  if (branch_name_to_input_control_num_.find(current_branch_name_) == branch_name_to_input_control_num_.end()) {
    input_controls_num_ = 0;
  } else {
    input_controls_num_ = branch_name_to_input_control_num_[current_branch_name_];
  }
  if (input_datas_num_ == 0 && input_controls_num_ == 0) {
    MS_LOG(EXCEPTION) << "No input data and input control, branch id:" << current_branch_name_
                      << " for actor:" << GetAID();
  }
  MS_LOG(DEBUG) << "Input data num:" << input_datas_num_ << " control num:" << input_controls_num_
                << " for actor:" << GetAID();
}

void ConditionGatherActor::ExecuteInferShapeTask(OpContext<DeviceTensor> *const context) {
  ExecuteLaunchKernelTask(context);
}

void ConditionGatherActor::ExecuteResizeKernelModTask(OpContext<DeviceTensor> *const context) {}

void ConditionGatherActor::ExecuteLaunchKernelTask(OpContext<DeviceTensor> *const context) {
  new_memory_free_list_.clear();
  for (size_t i = 0; i < branch_names_.size(); ++i) {
    branch_flags_.get()[i] = false;
  }
  if (input_device_tensors_.size() != output_device_tensors_.size() * branch_names_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input tensor size:" << input_device_tensors_.size()
                      << " and output device tensor size:" << output_device_tensors_.size()
                      << " branch name size:" << branch_names_ << " for actor:" << GetAID();
  }
  // Current branch name is set by the condition switch actor, it is used to make sure the real output index.
  auto iter = std::find(branch_names_.begin(), branch_names_.end(), current_branch_name_);
  if (iter == branch_names_.end()) {
    MS_LOG(EXCEPTION) << "Invalid branch name :" << current_branch_name_ << " all branch name:" << branch_names_.size()
                      << " for actor:" << GetAID();
  }
  size_t index = LongToSize(iter - branch_names_.begin());

  // Collect the device address should be freed.
  for (size_t input_index : input_free_index_) {
    if (input_index < index * output_device_tensors_.size() ||
        input_index >= (index + 1) * output_device_tensors_.size()) {
      continue;
    }
    if (input_device_tensors_[input_index] == nullptr) {
      MS_LOG(EXCEPTION) << "Failed to get input device tensor index:" << input_index
                        << " for node:" << kernel_->DebugString() << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(input_device_tensors_[input_index]);
    MS_LOG(DEBUG) << "Add decrease new ref count for device address:" << input_device_tensors_[input_index]
                  << " in actor:" << GetAID();
  }

  for (size_t output_index : output_free_index_) {
    if (output_index >= output_device_tensors_.size() || output_device_tensors_[output_index] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid output device tensor index:" << output_index
                        << "total size:" << output_device_tensors_.size() << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(output_device_tensors_[output_index]);
    MS_LOG(DEBUG) << "Add decrease new ref count for device address:" << output_device_tensors_[output_index]
                  << " in actor:" << GetAID();
  }
  if (new_memory_free_list_.size() > 0) {
    SendMemoryFreeReq(context);
  }
}

void ConditionGatherActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumOne) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  MS_EXCEPTION_IF_NULL(kernel_);
  if (!kernel_->HasAttr(kAttrBranchOutputNum)) {
    MS_LOG(EXCEPTION) << "Failed to get branch output num by actor:" << GetAID();
  }
  const auto &output_value = kernel_->GetAttr(kAttrBranchOutputNum);
  MS_EXCEPTION_IF_NULL(output_value);
  branch_output_num_ = GetValue<size_t>(output_value);
  MS_LOG(DEBUG) << "branch output num:" << branch_output_num_ << " for actor:" << GetAID();

  if (!kernel_->HasAttr(kAttrBranchGraphName)) {
    MS_LOG(EXCEPTION) << "Failed to get inline graph name by actor:" << GetAID();
  }
  const auto &branch_graph_names = kernel_->GetAttr(kAttrBranchGraphName);
  MS_EXCEPTION_IF_NULL(branch_graph_names);
  MS_LOG(DEBUG) << "Branch graph name:" << branch_graph_names->ToString() << " for actor:" << GetAID();
  if (!branch_graph_names->isa<ValueTuple>()) {
    MS_LOG(EXCEPTION) << "Invalid branch group name:" << branch_graph_names->ToString() << " for actor:" << GetAID();
  }
  const auto &tuple_name = branch_graph_names->cast<ValueTuplePtr>();
  MS_EXCEPTION_IF_NULL(tuple_name);
  std::for_each(tuple_name->value().begin(), tuple_name->value().end(),
                [this](const auto &value) { branch_names_.emplace_back(GetValue<std::string>(value)); });
  MS_LOG(DEBUG) << "Branch names:" << branch_names_ << " for actor:" << GetAID();

  size_t input_num = branch_output_num_ * branch_names_.size();
  input_device_tensors_.resize(input_num);
  pre_input_device_tensors_.resize(input_num);
  input_kernel_tensors_.resize(input_num);
  input_kernel_tensors_for_infer_.resize(input_num);
  memory_free_list_.resize(input_num);

  kernel_info_ = dynamic_cast<KernelInfo *>(kernel_->kernel_info());
  MS_EXCEPTION_IF_NULL(kernel_info_);
  kernel_mod_ = kernel_info_->MutableKernelMod();
  MS_EXCEPTION_IF_NULL(kernel_mod_);
  const auto &output_addresses = kernel_info_->output_address_list();
  const auto &somas_outputs = kernel_info_->somas_output_result();
  if (output_addresses.size() != somas_outputs.size()) {
    MS_LOG(DEBUG) << "Invalid output address size:" << output_addresses.size()
                  << " and somas output size:" << somas_outputs.size() << " for actor:" << GetAID();
  }
  for (size_t i = 0; i < output_addresses.size(); ++i) {
    auto &output_address = output_addresses[i];
    MS_EXCEPTION_IF_NULL(output_address);
    if (output_address->stream_id() != kernel_info_->stream_id()) {
      MS_LOG(DEBUG) << "Output address : " << output_address << " stream id :" << output_address->stream_id()
                    << " is not equal kernel info stream id : " << kernel_info_->stream_id() << ".";
    }
    (void)output_device_tensors_.emplace_back(output_address.get());
    (void)output_kernel_tensors_.emplace_back(output_address->kernel_tensor().get());
    // The output taken over by soma does not need to allocate memory.
    if (kernel_info_->IsTensorEnableSomas(somas_outputs, i)) {
      // Somas outputs use the info of kernelMod, and output address use the info of device address.
      if (somas_outputs[i].second < output_address->GetSize()) {
        MS_LOG(DEBUG) << GetAID().Name() << " check somas size warning, output index:" << i
                      << " somas aligned size:" << somas_outputs[i].second
                      << " is smaller than address size:" << output_address->GetSize();
      }
      // Used to keep graph output address when somas block memory free, and reused by the ref conut in other graphs.
      if (somas_graph_output_indexes_.count(i) > 0) {
        MS_LOG(DEBUG) << "Somas keep output device address:" << output_address << " ptr:" << output_address->GetPtr();
        MS_EXCEPTION_IF_NULL(somas_info_);
        (void)somas_info_->InsertGraphOutputInfo(output_address.get(), somas_outputs[i].first, somas_outputs[i].second);
        output_address->set_from_mem_pool(true);
        need_clean_ptr_device_addresses_.emplace_back(output_address);
      }
    }
  }

  for (size_t i = 0; i < input_num; ++i) {
    const auto &input_device_tensor = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel_, i, false);
    MS_EXCEPTION_IF_NULL(input_device_tensor);
    (void)real_input_data_infos_.emplace_back(
      std::make_shared<InputDataInfo>(input_device_tensor->format(), input_device_tensor->host_shape(),
                                      input_device_tensor->GetSize(), input_device_tensor->type_id()));
  }

  for (size_t index : input_free_index_) {
    if (index >= input_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << index << " output size:" << input_device_tensors_.size()
                        << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(input_device_tensors_[index]);
  }
  for (size_t index : output_free_index_) {
    if (index >= output_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Invalid output index:" << index << " output size:" << output_device_tensors_.size()
                        << " for actor:" << GetAID();
    }
    new_memory_free_list_.emplace_back(output_device_tensors_[index]);
  }

  if (output_device_tensors_.size() * branch_names_.size() != input_device_tensors_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input tensor size:" << input_device_tensors_.size()
                      << " branch size:" << branch_names_.size() << " and output size:" << output_device_tensors_.size()
                      << " for actor:" << GetAID();
  }
}

void ConditionGatherActor::UpdateRefDeviceAddress(OpContext<DeviceTensor> *const context, bool increase_ref_count) {
  if (input_device_tensors_.size() != output_device_tensors_.size() * branch_names_.size()) {
    MS_LOG(EXCEPTION) << "Invalid input tensor size:" << input_device_tensors_.size()
                      << " and output device tensor size:" << output_device_tensors_.size()
                      << " branch name size:" << branch_names_ << " for actor:" << GetAID();
  }
  auto iter = std::find(branch_names_.begin(), branch_names_.end(), current_branch_name_);
  if (iter == branch_names_.end()) {
    MS_LOG(EXCEPTION) << "Invalid branch name :" << current_branch_name_ << " all branch name:" << branch_names_.size()
                      << " for actor:" << GetAID();
  }

  // Actor output should be ref to the current branch.
  size_t index = LongToSize(iter - branch_names_.begin());
  for (size_t i = 0; i < output_device_tensors_.size(); ++i) {
    size_t input_index = i + index * output_device_tensors_.size();
    if (input_device_tensors_[input_index] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid input device tensor index:" << input_index << " for actor:" << GetAID();
    }
    if (output_device_tensors_[i] == nullptr) {
      MS_LOG(EXCEPTION) << "Invalid input device tensor index:" << input_index << " for actor:" << GetAID();
    }

    const auto &somas_outputs = kernel_info_->somas_output_result();
    if (kernel_info_->IsTensorEnableSomas(somas_outputs, i) && i < somas_outputs.size() &&
        somas_outputs[i].second > 0 && somas_graph_output_indexes_.count(i) > 0) {
      MS_LOG(DEBUG) << "Skip set ref for output index:" << i << " for actor:" << GetAID();
      continue;
    }
    output_device_tensors_[i]->set_tensor_storage_info(input_device_tensors_[input_index]->GetTensorStorageInfo());
    output_device_tensors_[i]->set_pointer_ref_count(input_device_tensors_[input_index]->pointer_ref_count());
    output_device_tensors_[i]->IncreaseNewRefCount(GetAID().Name());
    MS_LOG(DEBUG) << "Actor:" << GetAID() << " increase new ref count:" << output_device_tensors_[i]->new_ref_count()
                  << " and set ref device address:" << output_device_tensors_[i]->PrintInfo()
                  << " ref input device address:" << input_device_tensors_[input_index]->PrintInfo();
  }
  new_memory_free_list_.resize(input_free_index_.size() + output_free_index_.size());
}

void ConditionGatherActor::FetchParameterInput(size_t start_index, OpContext<DeviceTensor> *const context) {
  // Fetch parameter input tensor from device tensor store.
  if (!enable_input_optimize_) {
    return;
  }

  for (auto &parameter_index : parameter_indexs_) {
    if (parameter_index.first < start_index || parameter_index.first - start_index >= input_device_tensors_.size()) {
      continue;
    }
    auto device_tensor = FetchParameter(parameter_index.second, context, device_contexts_[0], GetAID());
    if (device_tensor == nullptr) {
      std::string error_info =
        GetAID().Name() + " get graph parameter store failed: " + parameter_index.second.first.first->DebugString() +
        ", device type:" + std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_device_tensors_[parameter_index.first - start_index] = device_tensor;
  }
}

void ConditionGatherActor::FetchInput(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  auto iter = std::find(branch_names_.begin(), branch_names_.end(), current_branch_name_);
  if (iter == branch_names_.end()) {
    MS_LOG(EXCEPTION) << "Invalid current branch name:" << current_branch_name_ << " total:" << branch_names_
                      << " for actor:" << GetAID();
  }
  size_t start_index = branch_output_num_ * LongToSize(iter - branch_names_.begin());

  memory_free_list_.clear();
  // Fetch input device tensor from input data.
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  if (data_iter != input_op_datas_.end()) {
    for (auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      if (IntToSize(input_data->index_) < start_index ||
          IntToSize(input_data->index_) - start_index >= input_device_tensors_.size()) {
        std::string error_info =
          "Invalid input index:" + std::to_string(input_data->index_) + " start:" + std::to_string(start_index) +
          " total:" + std::to_string(input_device_tensors_.size()) + " for actor:" + GetAID().Name();
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
      }
      MS_EXCEPTION_IF_NULL(input_data->data_);
      input_device_tensors_[IntToSize(input_data->index_) - start_index] = input_data->data_;

      memory_free_list_.emplace_back(input_data->data_);
    }
  }

  // Fetch input device tensor from device tensor store.
  for (auto &device_tensor_store_key : device_tensor_store_keys_) {
    if (device_tensor_store_key.first < start_index ||
        device_tensor_store_key.first - start_index >= input_device_tensors_.size()) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(device_tensor_store_key.second);
    auto device_tensor = DeviceTensorStore::GetInstance().Fetch(device_tensor_store_key.second.get(),
                                                                device_contexts_[0]->GetDeviceType());
    if (device_tensor == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_key.second->DebugString() +
        ", device type:" + std::to_string(static_cast<int>(device_contexts_[0]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    input_device_tensors_[device_tensor_store_key.first - start_index] = device_tensor.get();
  }

  FetchParameterInput(start_index, context);

  if (output_data_.size() != output_data_arrows_.size()) {
    MS_LOG(EXCEPTION) << "Invalid output data size:" << output_data_.size()
                      << " and output data arrow size:" << output_data_arrows_.size() << " for actor:" << GetAID();
  }

  for (size_t i = 0; i < output_data_arrows_.size(); ++i) {
    MS_EXCEPTION_IF_NULL(output_data_arrows_[i]);
    MS_EXCEPTION_IF_NULL(output_data_[i].first);
    const auto &from_index = output_data_arrows_[i]->from_output_index_;
    if (IntToSize(from_index) >= input_device_tensors_.size() || from_index < 0) {
      MS_LOG(EXCEPTION) << "Invalid from index:" << from_index << " to actor:" << output_data_arrows_[i]->to_op_id_
                        << " to index:" << output_data_arrows_[i]->to_input_index_ << " for actor:" << GetAID();
    }
    if (input_device_tensors_[from_index] == nullptr) {
      std::string error_info =
        GetAID().Name() + " get input device tensor index:" + std::to_string(from_index) + " failed.";
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    output_data_[i].first->data_ = input_device_tensors_[from_index];
    if (output_device_tensors_[from_index]->from_mem_pool()) {
      input_device_tensors_[from_index]->set_from_mem_pool(true);
    }
  }
}

void ConditionGatherActor::Run(OpContext<DeviceTensor> *const context) {
  try {
    MS_EXCEPTION_IF_NULL(kernel_);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), kernel_->fullname_with_scope(),
                                                   kernel_->func_graph()->ToString(), false);
    FetchInput(context);
    if (memory_free_list_.size() > 0) {
      SendMemoryFreeReq(context);
    }
    MS_LOG(DEBUG) << "Launch kernel:" << kernel_->fullname_with_scope();
    EraseInput(context);
    for (const auto &device_address : output_device_tensors_) {
      device_address->set_ptr(nullptr);
    }
    SetSomasMemory(context);
    SendOutput(context);
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    std::string error_info =
      "#umsg#Kernel error:#umsg#run kernel[" + kernel_->fullname_with_scope() + "] failed, exception: " + e.what();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR_BY_STRATEGY(GraphExecutionStrategy::kPipeline, (*context), error_info);
  }
}
}  // namespace runtime
}  // namespace mindspore
