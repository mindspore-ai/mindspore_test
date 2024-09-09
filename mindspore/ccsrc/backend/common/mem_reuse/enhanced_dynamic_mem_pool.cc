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
#include "include/backend/mem_reuse/enhanced_dynamic_mem_pool.h"

#include <iostream>

#include "include/backend/mem_reuse/mem_tracker.h"
#include "utils/log_adapter.h"
#ifdef ENABLE_DEBUGGER
#include "plugin/device/cpu/hal/profiler/cpu_profiling.h"
#endif

namespace mindspore {
namespace device {
void EnhancedDynamicMemPool::ReleaseDeviceRes() {
  AbstractDynamicMemPool::ReleaseDeviceRes();
  tracker::MemTrackerManager::GetInstance().Dump();
}

DeviceMemPtr EnhancedDynamicMemPool::AllocTensorMem(size_t size, bool from_persistent_mem, bool need_recycle,
                                                    uint32_t stream_id) {
  size_t align_size = AlignMemorySize(size);
  MS_LOG(DEBUG) << "Allocate tensor mem, size : " << size << ", align_size : " << align_size
                << ", from_persistent_mem : " << from_persistent_mem << ", need_recycle : " << need_recycle
                << ", stream_id : " << stream_id << ".";
  LockGuard lock(lock_);
  const auto [mem_buf, allocator] = AbstractDynamicMemPool::AllocMemBuf(align_size, from_persistent_mem, stream_id);
  if (mem_buf == nullptr) {
    MS_LOG(DEBUG) << "Allocate tensor mem, return nullptr.";
    return nullptr;
  }

  mem_buf->SetDebugInfo();
  addr_mem_buf_allocators_.emplace(mem_buf->addr_, std::make_pair(mem_buf, allocator));
  auto device_addr = mem_buf->addr_;

#ifdef ENABLE_DEBUGGER
  // Cpu profiler record.
  static auto profiler_inst = profiler::cpu::CPUProfiler::GetInstance();
  MS_EXCEPTION_IF_NULL(profiler_inst);
  if (profiler_inst->GetEnableFlag() && profiler_inst->GetProfileMemoryFlag()) {
    profiler_inst->RecordMemoryPoolInfo(TotalUsedMemStatistics(), TotalMemStatistics(),
                                        TotalUsedByEventMemStatistics());
  }
#endif

  // Adapt for dry run.
  if (common::IsNeedProfileMemory()) {
    MS_LOG(WARNING) << "Need Profile Memory, Memory pool alloc, total mem: " << TotalMemStatistics()
                    << ", peak mem: " << UsedMemPeakStatistics() << ", in use mem: " << TotalUsedMemStatistics()
                    << ", used by event mem: " << TotalUsedByEventMemStatistics()
                    << ", device address addr: " << device_addr << ", size: " << align_size
                    << ", from persistent mem: " << from_persistent_mem << ", need recycle: " << need_recycle;
  }

  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    tracker::CALL_MEMORY_TRACKER(AllocMemBlock, device_addr, mem_buf->size_, GetMemoryPoolType(),
                                 ActualPeakStatistics(), TotalUsedMemStatistics(), TotalMemStatistics(), stream_id);
  }

  MS_LOG(DEBUG) << "Allocate tensor mem, return : " << device_addr << ", stat : " << mem_stat_.ToJson() << ".";
  return device_addr;
}

std::vector<DeviceMemPtr> EnhancedDynamicMemPool::AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                                           uint32_t stream_id) {
  MS_LOG(DEBUG) << "Alloc continuous tensor mem.";
  return AbstractDynamicMemPool::AllocContinuousTensorMem(size_list, stream_id);
}

void EnhancedDynamicMemPool::FreeTensorMem(const DeviceMemPtr &device_addr) {
  MS_LOG(DEBUG) << "Free tensor mem, device addr : " << device_addr << ".";
  AbstractDynamicMemPool::FreeTensorMem(device_addr);
}

bool EnhancedDynamicMemPool::DoFreeTensorMem(const DeviceMemPtr &device_addr) {
  void *enhanced_device_addr = device_addr;
  bool ret = AbstractDynamicMemPool ::DoFreeTensorMem(device_addr);
  if (ret) {
#ifdef ENABLE_DEBUGGER
    // Cpu profiler record.
    static auto profiler_inst = profiler::cpu::CPUProfiler::GetInstance();
    MS_EXCEPTION_IF_NULL(profiler_inst);
    if (profiler_inst->GetEnableFlag() && profiler_inst->GetProfileMemoryFlag()) {
      profiler_inst->RecordMemoryPoolInfo(TotalUsedMemStatistics(), TotalMemStatistics(),
                                          TotalUsedByEventMemStatistics());
    }
#endif

    // Adapt for dry run.
    if (common::IsNeedProfileMemory()) {
      MS_LOG(WARNING) << "Need Profile Memory, Memory pool free, total mem: " << TotalMemStatistics()
                      << ", peak mem: " << UsedMemPeakStatistics() << ", in use mem: " << TotalUsedMemStatistics()
                      << ", used by event mem: " << TotalUsedByEventMemStatistics()
                      << ", device address addr: " << enhanced_device_addr << ".";
    }

    // Adapt for mem tracker.
    if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
      tracker::CALL_MEMORY_TRACKER(FreeMemBlock, enhanced_device_addr, TotalUsedMemStatistics(), TotalMemStatistics());
    }
  }
  MS_LOG(DEBUG) << "Do free tensor mem : " << enhanced_device_addr << ", ret : " << ret << ".";
  return ret;
}

void EnhancedDynamicMemPool::FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                const std::vector<DeviceMemPtr> &keep_addrs,
                                                const std::vector<size_t> &keep_addr_sizes) {
  MS_LOG(DEBUG) << "Free part tensor mems.";
  LockGuard lock(lock_);
  const auto keep_mem_bufs = AbstractDynamicMemPool::DoFreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    for (const auto &mem_buf : keep_mem_bufs) {
      tracker::CALL_MEMORY_TRACKER(AllocMemBlock, mem_buf->addr_, mem_buf->size_, GetMemoryPoolType(),
                                   ActualPeakStatistics(), TotalUsedMemStatistics(), TotalMemStatistics(),
                                   mem_buf->stream_id_);
    }
  }
}

void EnhancedDynamicMemPool::DefragMemory() {
  if (last_vmm_used_size_ == 0) {
    last_vmm_used_size_ = GetVmmUsedMemSize();
  } else {
    size_t vmm_used_size = GetVmmUsedMemSize();
    if (vmm_used_size > last_vmm_used_size_) {
      MS_LOG(WARNING) << "Current vmm used size : " << vmm_used_size
                      << " is bigger than last vmm used size : " << last_vmm_used_size_ << ".";
      last_vmm_used_size_ = vmm_used_size;
    }
  }

  AbstractDynamicMemPool::DefragMemory();
}

void EnhancedDynamicMemPool::DumpDynamicMemPoolStateInfo() {
  const auto &state_info = DynamicMemPoolStateInfo();
  static bool is_enable_memory_statistics = common::IsEnableRuntimeConfig(common::kRuntimeMemoryStat);
  if (is_enable_memory_statistics) {
    std::cout << "[MS_RUNTIME_PROF]" << state_info << std::endl;
  }
  MS_LOG(INFO) << state_info;
}

const std::pair<size_t, size_t> EnhancedDynamicMemPool::FreeIdleMemsByEagerFree() {
  const auto [eager_free_size, real_free_size] = AbstractDynamicMemPool::FreeIdleMemsByEagerFree();
  static bool is_enable_memory_statistics = common::IsEnableRuntimeConfig(common::kRuntimeMemoryStat);
  if (is_enable_memory_statistics) {
    std::cout << "Total eager free memory : " << eager_free_size << ", real free : " << real_free_size << "."
              << std::endl;
  }
  return {eager_free_size, real_free_size};
}
}  // namespace device
}  // namespace mindspore
