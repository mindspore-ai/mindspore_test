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

#ifndef MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_CPU_CPU_MEM_MANAGER_CPU_MEMORY_MANAGER_H_
#define MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_CPU_CPU_MEM_MANAGER_CPU_MEMORY_MANAGER_H_
#include <vector>
#include <map>
#include <memory>
#include "ir/device_address.h"
#include "runtime/hardware_abstract/memory_manager/memory_manager.h"
#include "plugin/cpu/res_manager/mem_manager/cpu_memory_pool.h"

namespace mindspore {
namespace device {
namespace cpu {
class CPUMemoryManager : public MemoryManager {
 public:
  CPUMemoryManager() = default;
  virtual ~CPUMemoryManager();

  void Initialize() override {}
  void Finalize() override { CPUMemoryPool::GetInstance().ReleaseDeviceRes(); }
  void ResetDynamicMemory() override;

  void *StaticMemMalloc(size_t mem_size);
  void MemFree(void *ptr);

  void *MallocMemFromMemPool(size_t size, bool from_persistent_mem, bool need_recycle = false,
                             uint32_t stream_id = kDefaultStreamIndex) override {
    return CPUMemoryPool::GetInstance().AllocTensorMem(size, from_persistent_mem, false, stream_id);
  }
  void FreeMemFromMemPool(void *device_ptr) override { CPUMemoryPool::GetInstance().FreeTensorMem(device_ptr); }
  std::vector<void *> MallocContinuousMemFromMemPool(const std::vector<size_t> &size_list,
                                                     uint32_t stream_id = kDefaultStreamIndex) override {
    return CPUMemoryPool::GetInstance().AllocContinuousTensorMem(size_list, stream_id);
  }

  DynamicMemPool *GetMemoryPool() override {
    if (MS_UNLIKELY(memory_pool_ == nullptr)) {
      memory_pool_ = &(CPUMemoryPool::GetInstance());
    }
    return memory_pool_;
  }

  bool GetDynamicMalloc() { return dynamic_malloc_; }

 protected:
  uint8_t *MallocStaticMem(size_t size, bool communication_mem, uint32_t graph_id) override;
  uint8_t *MallocDynamicMem(size_t size, bool communication_mem) override;

 private:
  uint8_t *MemMalloc(size_t size);
  void MemFree() noexcept;

  size_t mem_size_{0};
  uint8_t *mem_ptr_{nullptr};
  bool dynamic_malloc_{false};
  std::map<void *, size_t> dynamic_mem_;
  std::map<void *, size_t> static_mem_;
  std::map<void *, size_t> cached_mem_;
  std::map<void *, std::shared_ptr<std::vector<uint8_t>>> mem_block_map_;
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_PLUGIN_RES_MANAGER_CPU_CPU_MEM_MANAGER_CPU_MEMORY_MANAGER_H_
