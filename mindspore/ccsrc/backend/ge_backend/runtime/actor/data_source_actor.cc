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

#include "backend/ge_backend/runtime/actor/data_source_actor.h"
#include "backend/ge_backend/runtime/actor/memory_manager_actor.h"
#include "backend/ge_backend/runtime/actor/output_actor.h"
#include "backend/ge_backend/runtime/actor/recorder_actor.h"
#include "backend/ge_backend/runtime/actor/debug_actor.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "utils/phase.h"
#include "common/common_utils.h"
#include "utils/ms_context.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "runtime/device/res_manager/hal_res_manager.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
void DataSourceActor::Init() { InitOutputData(); }

void DataSourceActor::FetchData(OpContext<DeviceTensor> *const context) {
  MS_LOG(INFO) << "Data source actor(" << GetAID().Name() << ") fetches data.";
  MS_EXCEPTION_IF_NULL(context);
  device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, GetAID().Name(), GetAID().Name(), "");
  // Pop the data of last time.
  if (!buffers_.empty()) {
    buffers_.pop();
  }

  // Construct device tensors and fill to the buffers from member nodes.
  FillDataBuffer();
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }

  // Allocate memory for device tensors.
  SendMemoryAllocReq(context);
}

void DataSourceActor::UpdateOutputData(OpData<DeviceTensor> *const output_data, const DataArrowPtr &data_arrow,
                                       const AnfNodePtr &output_node, OpContext<DeviceTensor> *const context) {
  MS_EXCEPTION_IF_NULL(output_data);
  MS_EXCEPTION_IF_NULL(data_arrow);
  MS_EXCEPTION_IF_NULL(output_node);
  MS_EXCEPTION_IF_NULL(context);

  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }
  const auto &output_device_tensors = buffers_.front();

  auto position = FetchNodePosition({output_node, data_arrow->from_output_index_});
  // Host data souruce actor uses the node position, device data source actor uses the output index.
  auto output_position = (position != 0) ? position : IntToSize(data_arrow->from_output_index_);
  if (output_position >= output_device_tensors.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The output index is of range.");
  }
  output_data->data_ = output_device_tensors[output_position];
}

void HostQueueDataSourceActor::FillDataBuffer() {
  // Construct device tensors.
  std::vector<DeviceTensor *> device_tensors;
  for (auto &node_with_index : data_node_with_indexs_) {
    MS_LOG(DEBUG) << "Node:" << node_with_index.first->DebugString() << " index:" << node_with_index.second;
    auto device_address = AnfAlgo::GetMutableOutputAddr(node_with_index.first, node_with_index.second, false);
    MS_EXCEPTION_IF_NULL(device_address);
    (void)device_tensors.emplace_back(device_address.get());
  }

  buffers_.push(device_tensors);
}

void HostQueueDataSourceActor::SendMemoryAllocReq(OpContext<DeviceTensor> *const context) {
  auto &device_tensors = buffers_.back();
  if (ActorDispatcher::is_memory_allocation_sync()) {
    if (IsSameDeviceType()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &device_tensors, context,
                                GetAID());
    } else {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::AllocateBatchMemory, &device_tensors, context,
                                GetAID());
    }
    OnMemoryAllocFinish(context);
  } else {
    if (IsSameDeviceType()) {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateMemory, &device_tensors, context,
                            GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::AllocateBatchMemory, &device_tensors, context,
                            GetAID());
    }
  }
}

void HostQueueDataSourceActor::SendMemoryFreeReq(OpContext<DeviceTensor> *const context) {
  auto &device_tensors = buffers_.front();
  if (ActorDispatcher::is_memory_free_sync()) {
    if (IsSameDeviceType()) {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &device_tensors, context,
                                GetAID());
    } else {
      ActorDispatcher::SendSync(memory_manager_aid_, &MemoryManagerActor::FreeBatchMemory, &device_tensors, context,
                                GetAID());
    }
  } else {
    if (IsSameDeviceType()) {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeMemory, &device_tensors, context, GetAID());
    } else {
      ActorDispatcher::Send(memory_manager_aid_, &MemoryManagerActor::FreeBatchMemory, &device_tensors, context,
                            GetAID());
    }
  }
}

void HostQueueDataSourceActor::AddCopyDataCallBack(bool enable_async_copy,
                                                   const mindspore::tensor::TensorPtrList &host_tensors,
                                                   const std::vector<DeviceTensor *> &device_tensors) {
  if (!enable_async_copy || device_tensors.empty()) {
    return;
  }

  std::function<void(void)> callback_func = [host_tensors]() {
    // Clear buffer automatically.
  };

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::ResKey res_key{device::GetDeviceTypeByName(device_name), device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);
  auto callback_ret = res_manager->LaunchCallback(callback_func, device_tensors[0]->stream_id());
  if (!callback_ret) {
    MS_LOG(EXCEPTION) << "Async Copy memory launch callback failed";
  }
}

void HostQueueDataSourceActor::OnMemoryAllocFinish(OpContext<DeviceTensor> *const context) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  MS_EXCEPTION_IF_NULL(context);
  if (IsRunningFailed(context)) {
    return;
  }
  if (buffers_.size() == 0) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "The data queue is empty.");
  }

  // Get host tensors from host queue and get device tensors from buffers.
  MS_EXCEPTION_IF_NULL(host_queue_);
  if (host_queue_->IsEmpty()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Host data queue is empty.");
  }
  auto &host_tensors = host_queue_->Pull();
  auto &device_tensors = buffers_.back();
  if (host_tensors.size() != device_tensors.size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context),
                                      "The length of host tensors is not equal to the length of device tensors.");
  }

  // Copy data from host tensor to device tensor.
  uint64_t start_time = 0;
  PROFILER_START(start_time);
  auto enable_async_copy = ms_context->IsEnableInferBoost() || is_infer_phase_;
  try {
    for (size_t i = 0; i < host_tensors.size(); ++i) {
      auto &host_tensor = host_tensors[i];
      auto &device_tensor = device_tensors[i];
      MS_EXCEPTION_IF_NULL(device_tensor);
      MS_EXCEPTION_IF_NULL(host_tensor);
      // No used device address need skip.
      if (TEST_FLAG(device_tensor->flag(), device::kDeviceAddressFlagNotUsed)) {
        MS_LOG(DEBUG) << GetAID().Name() << " input index " << i << " is not used.";
        continue;
      }
      auto tensor_device_address = std::dynamic_pointer_cast<DeviceTensor>(host_tensor->device_address());
      // Sync data from host_tensor_device_address to device_tensor.
      if (tensor_device_address != nullptr) {
        if (tensor_device_address.get() == device_tensor) {
          continue;
        }
        if (!Copy(device_tensor, tensor_device_address.get())) {
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Copy data failed.");
        }
        continue;
      }
      if (host_tensor->data_ptr() == nullptr && device_tensor->GetSize() == 0) {
        MS_LOG(INFO) << "Empty tuple sync";
        continue;
      }

      if (enable_async_copy) {
        MS_LOG(INFO) << "Index :" << i
                     << ", data_node_with_indexs_[i].first : " << data_node_with_indexs_[i].first->DebugString();
        if (!device_tensor->AsyncHostToDevice(LongToSize(host_tensor->data().nbytes()), host_tensor->data_type(),
                                              host_tensor->data_ptr()->data())) {
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "SyncHostToDevice failed.");
        }
      } else {
        if (!device_tensor->SyncHostToDevice(
              AnfAlgo::GetRuntimePaddingShape(data_node_with_indexs_[i].first, data_node_with_indexs_[i].second),
              LongToSize(host_tensor->data().nbytes()), host_tensor->data_type(),
              host_tensor->device_info().host_format_, host_tensor->data_ptr())) {
          SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "SyncHostToDevice failed.");
        }
      }

      if (IsDynamic(device_tensor->host_shape())) {
        device_tensor->set_host_shape(host_tensor->shape());
      }
    }
    AddCopyDataCallBack(enable_async_copy, host_tensors, device_tensors);
  } catch (const std::exception &e) {
    MsException::Instance().SetException();
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*context), "Host data source actor run exception.");
  }
  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kCopyData, GetAID().Name(), false);

  PostRun(context);
}

size_t HostQueueDataSourceActor::FetchNodePosition(const KernelWithIndex &data_node) const {
  MS_EXCEPTION_IF_NULL(data_node.first);
  const auto &iter = data_node_position_map_.find(data_node);
  if (iter == data_node_position_map_.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, data_node.first)
      << "Data node: " << data_node.first->DebugString() << " index:" << data_node.second << " is not exist.";
  }
  return iter->second;
}

KernelWithIndex HostQueueDataSourceActor::FetchNode(size_t node_position) const {
  if (node_position >= data_node_with_indexs_.size()) {
    MS_LOG(EXCEPTION) << "The position of node is out of range: " << node_position;
  }
  return data_node_with_indexs_[node_position];
}

bool HostQueueDataSourceActor::IsSameDeviceType() const { return true; }

void HostQueueDataSourceActor::ReleaseData() {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kOutputProcess, "DataSourceActorReleaseData");
  // The step end need free the host queue tensor.
  MS_EXCEPTION_IF_NULL(host_queue_);
  host_queue_->Pop();

  // The step end need release data node address.
  for (auto &data_node_with_index : data_node_with_indexs_) {
    if (!AnfAlgo::OutputAddrExist(data_node_with_index.first, data_node_with_index.second)) {
      continue;
    }
    auto old_address = AnfAlgo::GetMutableOutputAddr(data_node_with_index.first, data_node_with_index.second);
    MS_EXCEPTION_IF_NULL(old_address);
    if (old_address->GetPtr() == nullptr) {
      // The Address memory is already freed.
      continue;
    }
    // If the address from input tensor and the address is not used by runtime.
    if (old_address->original_ref_count() == SIZE_MAX && !old_address->is_ptr_persisted()) {
      device::ResKey res_key{device::GetDeviceTypeByName(old_address->device_name()), old_address->device_id()};
      auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
      MS_EXCEPTION_IF_NULL(res_manager);

      const auto &kernel_tensor = old_address->kernel_tensor();
      MS_EXCEPTION_IF_NULL(kernel_tensor);
      auto new_kernel_tensor = kernel_tensor->CloneKernelTensor();
      MS_EXCEPTION_IF_NULL(new_kernel_tensor);
      new_kernel_tensor->set_device_ptr(nullptr);

      auto new_address = res_manager->CreateDeviceAddress(new_kernel_tensor);
      MS_EXCEPTION_IF_NULL(new_address);
      MS_LOG(DEBUG) << "Create device tensor:" << new_address << " type:" << new_address->type_id()
                    << ", kernel tensor addr:" << new_kernel_tensor.get();
      new_address->set_original_ref_count(old_address->original_ref_count());
      new_address->ResetRefCount();
      new_address->set_flag(old_address->flag());
      auto [node, index] = old_address->GetNodeIndex();
      new_address->SetNodeIndex(node, index);
      AnfAlgo::SetOutputAddr(new_address, data_node_with_index.second, data_node_with_index.first.get());
      if (ref_device_tensors_.find(data_node_with_index) == ref_device_tensors_.end()) {
        continue;
      }
      for (const auto &device_tensor : ref_device_tensors_[data_node_with_index]) {
        if (device_tensor != nullptr) {
          MS_LOG(DEBUG) << "Set pointer ref count from device address:" << new_address << " to:" << device_tensor
                        << " for data source node:" << data_node_with_index.first->DebugString();
          device_tensor->set_pointer_ref_count(new_address->pointer_ref_count());
        }
      }
    }
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
