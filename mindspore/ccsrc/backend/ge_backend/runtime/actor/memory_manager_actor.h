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

#ifndef MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_MEMORY_MANAGER_ACTOR_H_
#define MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_MEMORY_MANAGER_ACTOR_H_

#include <vector>
#include <memory>
#include <string>
#include <set>
#include <mutex>

#include "backend/ge_backend/runtime/actor/actor_common.h"
#include "backend/ge_backend/runtime/device_tensor_store.h"

namespace mindspore {
namespace ge_backend {
namespace runtime {
using mindspore::runtime::ProfilerEvent;
using mindspore::runtime::ProfilerModule;
using mindspore::runtime::ProfilerRecorder;

// MemoryManagerActor need response to memory alloc and free quickly, so must bind single thread.
class MemoryManagerActor : public ActorBase {
 public:
  static std::shared_ptr<MemoryManagerActor> &GetInstance();
  ~MemoryManagerActor() override = default;

  // The process entry of memory alloc.
  void AllocateMemory(const std::vector<KernelTensorPtr> *alloc_list, OpContext<kernel::KernelTensor> *const op_context,
                      const AID &from_aid);

  void AllocateBatchMemory(const std::vector<KernelTensorPtr> *alloc_list,
                           OpContext<kernel::KernelTensor> *const op_context, const AID &from_aid);

  // The process entry of memory free.
  void FreeMemory(const std::vector<KernelTensorPtr> *free_list, OpContext<kernel::KernelTensor> *const op_context,
                  const AID &from_aid);

  void FreeBatchMemory(const std::vector<KernelTensorPtr> *free_list, OpContext<kernel::KernelTensor> *const op_context,
                       const AID &from_aid);

  void FreeMemoryByRefCount(const KernelTensorPtr &kernel_tensor, const std::string &op_name);

  // Wait the MemoryManagerActor to finish running all current messages.
  void Wait(OpContext<kernel::KernelTensor> *const op_context, const AID &from_aid);

 private:
  MemoryManagerActor() : ActorBase("GEMemoryManagerActor") {}
  DISABLE_COPY_AND_ASSIGN(MemoryManagerActor);

  // When allocate device memory fail, print error log and set op context failed status.
  void SetOpContextMemoryAllocFail(const std::string &kernel_name, size_t alloc_size,
                                   OpContext<kernel::KernelTensor> *const op_context);

  // MemoryManagerActor object is used like a single instance, if one actor allocates memory failed in one batch, which
  // will set fail message info OpContext, major thread will destroy the OpContext object, subsequent actor can not set
  // fail message again, so we record allocating memory fail event by the uuid of the batch, which is key of the set.
  std::set<int> mem_alloc_failed_step_ids_;
  std::mutex mem_alloc_failed_mutex_;
};
}  // namespace runtime
}  // namespace ge_backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GEBACKEND_RUNTIME_ACTOR_MEMORY_MANAGER_ACTOR_H_
