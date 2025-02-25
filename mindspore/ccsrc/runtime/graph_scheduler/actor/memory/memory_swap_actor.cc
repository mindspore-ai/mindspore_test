/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "runtime/graph_scheduler/actor/memory/memory_swap_actor.h"

#include <map>

#include "runtime/device/res_manager/loadable_device_address.h"
#include "runtime/graph_scheduler/device_tensor_store.h"

namespace mindspore {
namespace runtime {
void MemorySwapActor::UpdateDeviceTensors(OpContext<mindspore::runtime::DeviceTensor> *context) {
  MS_EXCEPTION_IF_NULL(context);
  const auto &data_iter = input_op_datas_.find(context->sequential_num_);
  size_t total_device_tensor_num = fixed_device_tensor_num_ + device_tensor_store_keys_.size();
  if (data_iter != input_op_datas_.end()) {
    total_device_tensor_num += data_iter->second.size();
  }
  if (device_tensors_to_swap_.size() < total_device_tensor_num) {
    device_tensors_to_swap_.resize(total_device_tensor_num);
  }
  if (data_iter != input_op_datas_.end()) {
    for (const auto &input_data : data_iter->second) {
      MS_EXCEPTION_IF_NULL(input_data);
      size_t input_index = IntToSize(input_data->index_);
      const size_t swap_device_tensor_index = input_index + fixed_device_tensor_num_;
      if (swap_device_tensor_index >= total_device_tensor_num) {
        SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The input index is out of range.");
      }
      device_tensors_to_swap_[swap_device_tensor_index] = input_data->data_;
    }
  }
  for (const auto &key : device_tensor_store_keys_) {
    auto device_tensor = DeviceTensorStore::GetInstance().Fetch(key.second.get(), device_contexts_[0]->GetDeviceType());
    device_tensors_to_swap_[key.first] = device_tensor.get();
  }
}

std::vector<DeviceTensor *> MemorySwapActor::GetDeviceTensors(const std::vector<size_t> &indexes) {
  std::vector<DeviceTensor *> device_tensors;
  for (const auto index : indexes) {
    if (index >= device_tensors_to_swap_.size()) {
      MS_LOG(EXCEPTION) << "Device tensor index[" << index << "] out of range[" << device_tensors_to_swap_.size()
                        << "].";
    }
    (void)device_tensors.emplace_back(device_tensors_to_swap_[index]);
  }
  return device_tensors;
}

void MemorySwapActor::AllocDeviceContinuousMem(const std::vector<DeviceTensor *> &device_tensors) {
  std::vector<size_t> size_list;
  for (const auto device_tensor : device_tensors) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    size_list.emplace_back(device_tensor->GetSize());
  }
  MS_EXCEPTION_IF_CHECK_FAIL((!device_contexts_.empty()), "The device context doesn't exist.");
  MS_EXCEPTION_IF_NULL(device_contexts_[0]);
  MS_EXCEPTION_IF_NULL(device_contexts_[0]->device_res_manager_);
  const auto &device_ptrs = device_contexts_[0]->device_res_manager_->AllocateContinuousMemory(size_list);
  for (size_t i = 0; i < device_tensors.size(); ++i) {
    MS_EXCEPTION_IF_NULL(device_tensors[i]);
    // If data already exists in device, copy it to continuous memory and release the original data.
    if (device_tensors[i]->status() == device::DeviceAddressStatus::kInDevice &&
        device_tensors[i]->GetPtr() != nullptr) {
      const auto original_ptr = device_tensors[i]->GetMutablePtr();
      device_tensors[i]->set_ptr(device_ptrs[i]);
      if (!device_tensors[i]->SyncDeviceToDevice(device_tensors[i]->host_shape(), device_tensors[i]->GetSize(),
                                                 device_tensors[i]->type_id(), original_ptr,
                                                 device_tensors[i]->format())) {
        MS_LOG(EXCEPTION) << "Copy data for continuous memory failed, src addr: " << original_ptr << ", dst addr: "
                          << ", size: " << device_tensors[i]->GetSize();
      }
      device_contexts_[0]->device_res_manager_->FreeMemory(original_ptr);
    } else {
      device_tensors[i]->set_ptr(device_ptrs[i]);
    }
    device_tensors[i]->set_from_mem_pool(true);
  }
}

void MemorySwapActor::Swap(OpContext<mindspore::runtime::DeviceTensor> *const context, device::StorageType to,
                           const std::vector<DeviceTensor *> &device_tensors) {
  for (const auto &device_tensor : device_tensors) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    if (!device_tensor->MoveTo(to, false, kDefaultStreamIndex)) {
      SET_OPCONTEXT_FAIL_RET_WITH_ERROR(*context, "Swap tensor failed.");
    }
  }
}

void MemorySwapActor::Run(OpContext<mindspore::runtime::DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(context);
  static std::map<device::SwapActionType, device::StorageType> swap_to_map = {
    {device::SwapActionType::kHBM2DDR, device::StorageType::kHost},
    {device::SwapActionType::kDDR2HBM, device::StorageType::kDevice},
    {device::SwapActionType::kDDR2DISK, device::StorageType::kFile},
    {device::SwapActionType::kDISK2DDR, device::StorageType::kHost},
    {device::SwapActionType::kHBM2DISK, device::StorageType::kFile},
    {device::SwapActionType::kDISK2HBM, device::StorageType::kDevice}};
  UpdateDeviceTensors(context);
  for (const auto &action : swap_actions_) {
    const auto action_type = action.first;
    const auto &device_tensor_indexes = action.second;
    const auto &device_tensors = GetDeviceTensors(device_tensor_indexes);
    if (action_type == device::SwapActionType::kAllocHBM) {
      AllocDeviceContinuousMem(device_tensors);
    } else if (action_type != device::SwapActionType::kUnDefined) {
      Swap(context, swap_to_map[action_type], device_tensors);
    } else {
      MS_LOG(WARNING) << "Unknown swap action type, skip.";
    }
  }
  EraseInput(context);
  SendOutput(context);
}
}  // namespace runtime
}  // namespace mindspore
