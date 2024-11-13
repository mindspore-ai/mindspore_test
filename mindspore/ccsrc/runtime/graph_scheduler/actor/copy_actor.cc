/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/copy_actor.h"

#include <utility>
#include "runtime/graph_scheduler/actor/memory_manager_actor.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "include/backend/mem_reuse/mem_tracker.h"

namespace mindspore {
namespace runtime {
const size_t kInputDeviceContextIndex = 0;
const size_t kOutputDeviceContextIndex = 1;

void CopyActor::Init() {
  // Check device contexts number.
  if (device_contexts_.size() != device::kDeviceContextsNumTwo) {
    MS_LOG(EXCEPTION) << "The device contexts number is wrong.";
  }

  const size_t kDeviceTensorNum = 1;
  input_device_tensor_.resize(kDeviceTensorNum);
  output_device_tensor_.resize(kDeviceTensorNum);

  // Check output data index.
  for (auto &data_arrow : output_data_arrows_) {
    MS_EXCEPTION_IF_NULL(data_arrow);
    if (IntToSize(data_arrow->from_output_index_) != 0) {
      MS_LOG(EXCEPTION) << "The output index is out of range: " << GetAID().Name();
    }
  }

  InitOutputData();
}

void CopyActor::Run(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), GetAID().Name(), "");
  FetchDeviceTensor(context);
  SendMemoryAllocReq(context);
}

void CopyActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  if (ActorDispatcher::is_memory_allocation_sync()) {
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &output_device_tensor_,
                              device_contexts_[kOutputDeviceContextIndex], context, GetAID());
    OnMemoryAllocFinish(context);
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &output_device_tensor_,
                          device_contexts_[kOutputDeviceContextIndex], context, GetAID());
  }
}

void CopyActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  if (ActorDispatcher::is_memory_free_sync()) {
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &input_device_tensor_,
                              device_contexts_[kInputDeviceContextIndex], context, GetAID());
    ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &output_device_tensor_,
                              device_contexts_[kOutputDeviceContextIndex], context, GetAID());
  } else {
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &input_device_tensor_,
                          device_contexts_[kInputDeviceContextIndex], context, GetAID());
    ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &output_device_tensor_,
                          device_contexts_[kOutputDeviceContextIndex], context, GetAID());
  }
}

void UpdateRefDeviceTensors(const std::vector<DeviceTensor *> &input_device_tensor,
                            const std::vector<DeviceTensor *> &output_device_tensor,
                            const std::set<DeviceTensorPtr> &ref_parameter_device_tensors, const AID &aid) {
  MS_LOG(DEBUG) << "Add device tensor copy store for device address:" << output_device_tensor[0]
                << " type:" << output_device_tensor[0]->GetDeviceType() << " and " << input_device_tensor[0]
                << " type:" << input_device_tensor[0]->GetDeviceType() << " for copy actor:" << aid;
  DeviceTensorCopyStore::GetInstance().Insert(output_device_tensor[0], input_device_tensor[0]);
  for (const auto &device_tensor : ref_parameter_device_tensors) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    MS_LOG(DEBUG) << "Add device tensor copy store for device address:" << output_device_tensor[0]
                  << " type:" << output_device_tensor[0]->GetDeviceType() << " and " << device_tensor
                  << " type:" << device_tensor->GetDeviceType() << " for copy actor:" << aid;
    DeviceTensorCopyStore::GetInstance().Insert(output_device_tensor[0], device_tensor.get());
    if (device_tensor->GetDeviceType() != output_device_tensor[0]->GetDeviceType()) {
      MS_LOG(WARNING) << "Invalid ref device address:" << device_tensor << " type:" << device_tensor->GetDeviceType()
                      << " and:" << output_device_tensor[0] << " type:" << output_device_tensor[0]->GetDeviceType();
      continue;
    }
    device_tensor->set_ptr(output_device_tensor[0]->GetMutablePtr());
    MS_LOG(DEBUG) << "Set ptr:" << device_tensor->GetPtr() << " from device address:" << output_device_tensor[0]
                  << " to:" << device_tensor << " in actor:" << aid;
  }
}

bool CheckNonWeightParameter(const std::vector<std::pair<size_t, AnfNodePtr>> &device_tensor_store_keys,
                             const std::vector<std::pair<size_t, ParameterInfo>> &parameter_indexs) {
  if (EnableInputOptimize()) {
    if (parameter_indexs.size() > 0 && parameter_indexs[0].second.first.first->isa<Parameter>() &&
        common::AnfAlgo::IsParameterWeight(parameter_indexs[0].second.first.first->cast<ParameterPtr>())) {
      return false;
    }
    return true;
  }
  return device_tensor_store_keys.empty();
}

void CopyActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  MS_EXCEPTION_IF_NULL(output_device_tensor_[0]);
  MS_EXCEPTION_IF_NULL(input_device_tensor_[0]);
  if (IsRunningFailed(context)) {
    return;
  }

  if (input_device_tensor_[0]->GetSize() != output_device_tensor_[0]->GetSize()) {
    MS_LOG(WARNING) << GetAID().Name() << " copy size is not equal, input device tensor:" << input_device_tensor_[0]
                    << " size:" << input_device_tensor_[0]->GetSize()
                    << ", output device tensor:" << output_device_tensor_[0]
                    << "size:" << output_device_tensor_[0]->GetSize();
  }

  {
    ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kCopyData, GetAID().Name());
    MS_LOG(DEBUG) << "Copy device tensor from device address:" << input_device_tensor_[0]
                  << " type:" << input_device_tensor_[0]->GetDeviceType() << " to " << output_device_tensor_[0]
                  << " type:" << output_device_tensor_[0]->GetDeviceType() << " for copy actor:" << GetAID();
    if (!Copy(output_device_tensor_[0], input_device_tensor_[0])) {
      std::string error_info = "Copy device tensor failed: " + GetAID().Name();
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
    // Record ref map for non weight parameter.
    if (CheckNonWeightParameter(device_tensor_store_keys_, parameter_indexs_)) {
      UpdateRefDeviceTensors(input_device_tensor_, output_device_tensor_, ref_parameter_device_tensors_, GetAID());
    }
    output_device_tensor_[0]->kernel_tensor()->SetType(input_device_tensor_[0]->kernel_tensor()->GetType());
    output_device_tensor_[0]->kernel_tensor()->SetShape(input_device_tensor_[0]->kernel_tensor()->GetShape());
    output_device_tensor_[0]->set_user_data(input_device_tensor_[0]->user_data());
    MS_LOG(DEBUG) << "Set user data:" << input_device_tensor_[0]->user_data()
                  << " shape:" << input_device_tensor_[0]->kernel_tensor()->GetShape()->ToString()
                  << " from device tensor:" << input_device_tensor_[0]
                  << " to device address:" << output_device_tensor_[0];
    output_device_tensor_[0]->set_need_sync_user_data(input_device_tensor_[0]->need_sync_user_data());
  }

  PostRun(context);
}

void CopyActor::FetchParameterInput(OpContext<DeviceTensor> *const context) {
  if (!enable_input_optimize_) {
    return;
  }
  if (parameter_indexs_.size() > 0) {
    input_device_tensor_[0] =
      FetchParameter(parameter_indexs_[0].second, context, device_contexts_[kInputDeviceContextIndex], GetAID());
    if (input_device_tensor_[0] == nullptr) {
      std::string error_info =
        GetAID().Name() +
        " get graph parameter store input failed: " + parameter_indexs_[0].second.first.first->fullname_with_scope() +
        ", device type:" +
        std::to_string(static_cast<int>(device_contexts_[kInputDeviceContextIndex]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    if (parameter_indexs_[0].second.first.first->isa<Parameter>() &&
        !common::AnfAlgo::IsParameterWeight(parameter_indexs_[0].second.first.first->cast<ParameterPtr>())) {
      // Get non-weight parameter addr.
      MS_EXCEPTION_IF_NULL(output_);
      output_device_tensor_[0] = output_.get();
    } else {
      // Get weight parameter addr.
      output_device_tensor_[0] =
        FetchParameter(parameter_indexs_[0].second, context, device_contexts_[kOutputDeviceContextIndex], GetAID());
    }

    if (output_device_tensor_[0] == nullptr) {
      std::string error_info =
        GetAID().Name() +
        " get graph parameter store output failed: " + parameter_indexs_[0].second.first.first->fullname_with_scope() +
        ", device type:" +
        std::to_string(static_cast<int>(device_contexts_[kOutputDeviceContextIndex]->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  }
}

void CopyActor::FetchDeviceTensor(OpContext<DeviceTensor> *const context) {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kPreLaunch, GetAID().Name());
  MS_EXCEPTION_IF_NULL(context);
  const auto &input_device_context = device_contexts_[kInputDeviceContextIndex];
  const auto &output_device_context = device_contexts_[kOutputDeviceContextIndex];
  MS_EXCEPTION_IF_NULL(input_device_context);
  MS_EXCEPTION_IF_NULL(output_device_context);

  if (device_tensor_store_keys_.size() > 0) {
    const auto &device_tensor_store_node = device_tensor_store_keys_[0].second;
    MS_EXCEPTION_IF_NULL(device_tensor_store_node);
    input_device_tensor_[0] = DeviceTensorStore::GetInstance()
                                .Fetch(device_tensor_store_node.get(), input_device_context->GetDeviceType())
                                .get();
    if (input_device_tensor_[0] == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_node->fullname_with_scope() +
        ", device type:" + std::to_string(static_cast<int>(input_device_context->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }

    output_device_tensor_[0] = DeviceTensorStore::GetInstance()
                                 .Fetch(device_tensor_store_node.get(), output_device_context->GetDeviceType())
                                 .get();
    if (output_device_tensor_[0] == nullptr) {
      std::string error_info =
        GetAID().Name() + " get device tensor store failed: " + device_tensor_store_node->fullname_with_scope() +
        ", device type:" + std::to_string(static_cast<int>(output_device_context->GetDeviceType()));
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), error_info);
    }
  } else {
    if (input_op_datas_.empty()) {
      FetchParameterInput(context);
    } else {
      const auto &data_iter = input_op_datas_.find(context->sequential_num_);
      if (data_iter == input_op_datas_.end()) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "No input data.");
      }
      const auto &input_data = data_iter->second[0];
      MS_EXCEPTION_IF_NULL(input_data);
      input_device_tensor_[0] = input_data->data_;

      MS_EXCEPTION_IF_NULL(output_);
      output_device_tensor_[0] = output_.get();
    }
  }

  if (!WaitRuntimePipelineFinish(context)) {
    MS_LOG(INFO) << "Run failed and early stop.";
    return;
  }
  if (is_need_update_output_size_ && (input_device_tensor_[0]->GetSize() != output_device_tensor_[0]->GetSize())) {
    MS_LOG(DEBUG) << GetAID().Name() << " update output size from " << output_device_tensor_[0]->GetSize() << " to "
                  << input_device_tensor_[0]->GetSize();
    output_device_tensor_[0]->SetSize(input_device_tensor_[0]->GetSize());
    const auto &output_kernel_tensor = output_device_tensor_[0]->kernel_tensor();
    const auto &input_kernel_tensor = input_device_tensor_[0]->kernel_tensor();
    MS_EXCEPTION_IF_NULL(output_kernel_tensor);
    MS_EXCEPTION_IF_NULL(input_kernel_tensor);
    output_kernel_tensor->SetType(input_kernel_tensor->GetType()->Clone());
    output_kernel_tensor->SetShape(input_kernel_tensor->GetShape()->Clone());
  }
}

void CopyActor::UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &, const AnfNodePtr &,
                                 OpContext<DeviceTensor> *const) {
  MS_EXCEPTION_IF_NULL(output_data);
  output_data->data_ = output_device_tensor_[0];
}
}  // namespace runtime
}  // namespace mindspore
