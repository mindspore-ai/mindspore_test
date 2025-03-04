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

#include "backend/ge_backend/runtime/actor/memory_manager_actor.h"
#include "backend/ge_backend/runtime/actor/data_source_actor.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "async/async.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
namespace {
void OnMemoryAllocFinish(const AID &from_aid, OpContext<DeviceTensor> *const op_context) {
  if (!ActorDispatcher::is_memory_allocation_sync()) {
    ActorDispatcher::Send(from_aid, &MemoryAwareActor::OnMemoryAllocFinish, op_context);
  }
}
}  // namespace

void MemoryManagerActor::AllocateMemory(const std::vector<DeviceTensor *> *alloc_list,
                                        const DeviceContext *device_context, OpContext<DeviceTensor> *const op_context,
                                        const AID &from_aid) {
  for (auto &device_tensor : *alloc_list) {
    MS_EXCEPTION_IF_NULL(device_tensor);
    // Unused device address need skip to reduce memory use.
    if (device_tensor->IsNotNeedAlloc()) {
      continue;
    }

    if (device::tracker::MemTrackerManager::GetInstance().IsEnabled()) {
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddMemInfo, from_aid.Name(), memory::mem_pool::MemType::kKernel,
                                                     device_tensor->GetSize(), device_tensor);
    }

    try {
      // Allocate memory through the device context.
      bool success = device_context->device_res_manager_->AllocateMemory(device_tensor, kDefaultStreamIndex);
      if (!success) {
        SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
        return;
      }
    } catch (const std::exception &e) {
      SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
      return;
    }

    if (IsNeedProfilieMemoryLog()) {
      auto output_address = reinterpret_cast<std::uintptr_t>(device_tensor);
      MS_LOG(WARNING) << "Need Profile Memory, alloc type: MemoryManagerActor, device address class ptr: "
                      << output_address << ", device address size: " << device_tensor->GetSize()
                      << ", device address addr: " << device_tensor->GetPtr();
    }
  }
}

void MemoryManagerActor::AllocateBatchMemory(const std::vector<DeviceTensor *> *alloc_list,
                                             const std::vector<const DeviceContext *> *device_contexts,
                                             OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  MS_EXCEPTION_IF_NULL(alloc_list);
  MS_EXCEPTION_IF_NULL(device_contexts);
  MS_EXCEPTION_IF_NULL(op_context);
  if ((*alloc_list).size() != (*device_contexts).size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context),
                                      "The size of alloc list is not equal to the size of device contexts.");
  }

  for (size_t i = 0; i < (*alloc_list).size(); ++i) {
    auto &device_tensor = (*alloc_list)[i];
    auto &device_context = (*device_contexts)[i];
    MS_EXCEPTION_IF_NULL(device_tensor);
    MS_EXCEPTION_IF_NULL(device_context);
    // Unused device address need skip to reduce memory use.
    if (device_tensor->IsNotNeedAlloc()) {
      continue;
    }

    try {
      // Allocate memory through the device context.
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, from_aid.Name(), "BatchMemory", "");
      device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(
        AddMemInfo, from_aid.Name(), memory::mem_pool::MemType::kBatchMemory, device_tensor->GetSize(), device_tensor);
      if (!device_context->device_res_manager_->AllocateMemory(device_tensor, kDefaultStreamIndex)) {
        SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
        return;
      }
    } catch (const std::exception &e) {
      SetOpContextMemoryAllocFail(from_aid.Name(), device_context, device_tensor->GetSize(), op_context);
      return;
    }
  }

  // Call back to the from actor to process after memory allocation finished.
  OnMemoryAllocFinish(from_aid, op_context);

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryAlloc, from_aid.Name(), false);
}

void MemoryManagerActor::FreeMemory(const std::vector<DeviceTensor *> *free_list, const DeviceContext *device_context,
                                    OpContext<DeviceTensor> *, const AID &from_aid) {
  for (auto &device_tensor : *free_list) {
    FreeMemoryByRefCount(device_tensor, device_context, from_aid.Name());
  }
}

void MemoryManagerActor::FreeBatchMemory(const std::vector<DeviceTensor *> *free_list,
                                         const std::vector<const DeviceContext *> *device_contexts,
                                         OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  uint64_t start_time = 0;
  PROFILER_START(start_time);

  MS_EXCEPTION_IF_NULL(free_list);
  MS_EXCEPTION_IF_NULL(device_contexts);
  MS_EXCEPTION_IF_NULL(op_context);
  if ((*free_list).size() != (*device_contexts).size()) {
    SET_OPCONTEXT_FAIL_RET_WITH_ERROR((*op_context),
                                      "The size of free list is not equal to the size of device contexts.");
  }

  for (size_t i = 0; i < (*free_list).size(); ++i) {
    auto &device_tensor = (*free_list)[i];
    auto &device_context = (*device_contexts)[i];
    FreeMemoryByRefCount(device_tensor, device_context, from_aid.Name());
  }

  PROFILER_END(start_time, ProfilerModule::kRuntime, ProfilerEvent::kMemoryFree, from_aid.Name(), false);
}

void MemoryManagerActor::Wait(OpContext<DeviceTensor> *const op_context, const AID &from_aid) {
  // Call back to the from actor to process.
  ActorDispatcher::Send(from_aid, &MemoryAwareActor::OnMemoryAllocFinish, op_context);
}

// Only one of the static and dynamic reference counts will take effect.
void MemoryManagerActor::FreeMemoryByRefCount(DeviceTensor *const device_tensor, const DeviceContext *device_context,
                                              const std::string &op_name) {
  if (device_tensor == nullptr) {
    return;
  }
  if (device_tensor->original_ref_count() != SIZE_MAX) {
    // The static reference count is decremented to zero to free memory, and reset to the original count.
    size_t ref_count = device_tensor->DecreaseRefCount();
    if (ref_count == 0) {
      device_tensor->ResetRefCount();
      device_tensor->ClearUserData();
      if (device_tensor->GetPtr() != nullptr) {
        auto held_by_nodes = device_tensor->held_by_nodes();
        if (held_by_nodes.empty()) {
          FreeMemoryByDeviceContext(device_tensor, device_context);
        } else {
          FreeMemoryByValueNode(held_by_nodes, device_tensor);
        }
      }
    }
  } else if (device_tensor->dynamic_ref_count() != INT32_MAX) {
    // The dynamic reference count is decremented to zero to free memory.
    if ((device_tensor->DecreaseDynamicRefCount(op_name) == 0) && (device_tensor->GetPtr() != nullptr)) {
      device_tensor->ClearUserData();
      MS_LOG(DEBUG) << "Free memory by the dynamic reference count, device address" << device_tensor->GetPtr() << ".";
      if (device_tensor->deleter() != nullptr) {
        MS_LOG(DEBUG) << "Free ptr:" << device_tensor->GetPtr() << " for device address:" << device_tensor;
        device_tensor->deleter()(static_cast<uint8_t *>(device_tensor->GetMutablePtr()));
        device_tensor->set_deleter(nullptr);
        device_tensor->set_ptr(nullptr);
        return;
      }
      FreeMemoryByDeviceContext(device_tensor, device_context);
    }
  }
}

void MemoryManagerActor::SetOpContextMemoryAllocFail(const std::string &kernel_name,
                                                     const DeviceContext *device_context, size_t alloc_size,
                                                     OpContext<DeviceTensor> *const op_context) {
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(op_context);

  std::lock_guard<std::mutex> locker(mem_alloc_failed_mutex_);
  int step_id = op_context->sequential_num_;
  // First occur allocating memory failed.
  if (mem_alloc_failed_step_ids_.find(step_id) == mem_alloc_failed_step_ids_.end()) {
    mem_alloc_failed_step_ids_.clear();
    (void)mem_alloc_failed_step_ids_.insert(step_id);
    SET_OPCONTEXT_MEMORY_ALLOC_FAIL_BY_STRATEGY(GraphExecutionStrategy::kPipeline, *op_context, *device_context,
                                                kernel_name, alloc_size);
  }
}
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore
