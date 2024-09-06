/**
 * Copyright 2019-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_DYNAMIC_ALLOCATOR_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_DYNAMIC_ALLOCATOR_H_

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>
#include <random>
#include <string>
#include <tuple>

#include "utils/ms_utils.h"
#include "include/backend/mem_reuse/dynamic_mem_pool.h"
#include "include/backend/visible.h"
#include "include/common/utils/stream_util.h"
#include "ir/device_event.h"
#ifdef __APPLE__
#include "async/spinlock.h"
#endif

namespace mindspore {
namespace device {
// Implement of best fit dynamic memory pool.
class BACKEND_EXPORT DynamicMemPoolBestFit : virtual public DynamicMemPool {
 public:
  DynamicMemPoolBestFit()
      : persistent_mem_(std::make_shared<MemStatusManager>()), common_mem_(std::make_shared<MemStatusManager>()) {}
  virtual ~DynamicMemPoolBestFit();

  // The main program entry of memory alloc.
  DeviceMemPtr AllocTensorMem(size_t size, bool from_persistent_mem = false, bool need_recycle = false,
                              uint32_t stream_id = kDefaultStreamIndex) override;
  // The main program entry of continuous memory alloc.
  std::vector<DeviceMemPtr> AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                     uint32_t stream_id = kDefaultStreamIndex) override;
  // The main program entry of memory free.
  void FreeTensorMem(const DeviceMemPtr &device_addr) override;
  // The main program entry of part memorys free and part memorys keep.
  void FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs, const std::vector<DeviceMemPtr> &keep_addrs,
                          const std::vector<size_t> &keep_addr_sizes) override;

  // Release the real device memory.
  void ReleaseDeviceRes() override;

  // Get the minimum memory unit size using for dynamic extend.
  size_t MemAllocUnitSize(bool from_persistent_mem = false) const;
  // Set the minimum memory unit size using for dynamic extend.
  void SetMemAllocUintSize(size_t common_size, size_t persist_size = kDynamicMemAllocUnitSize);

  // Extract detailed block information
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> ExtractBlocksListInfo(
    const MemStatusManagerPtr &mem_mng) const;

  // The statistics information.
  size_t TotalMemStatistics() const override;
  size_t TotalUsedMemStatistics() const override;
  size_t TotalUsedByEventMemStatistics() const override;
  size_t TotalIdleMemStatistics() const override;
  size_t TotalEagerFreeMemStatistics() const override;
  size_t UsedMemPeakStatistics() const override;
  size_t MaxMemAllocatedStatistics() const override;
  size_t MaxMemReservedStatistics() const override;
  size_t ActualPeakStatistics() const override;
  std::unordered_map<std::string, std::size_t> BlockCountsStatistics() const override;
  std::unordered_map<std::string, std::size_t> BlockUnitSizeStatistics() const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> CommonMemBlocksInfoStatistics()
    const override;
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> PersistentMemBlocksInfoStatistics()
    const override;
  void ResetMaxMemReserved() override;
  void ResetMaxMemAllocated() override;

  // Display the brief state information of memory block and memory buf.
  void DumpDynamicMemPoolStateInfo() override;
  // Display the detailed debug information of memory block and memory buf.
  void DumpDynamicMemPoolDebugInfo() override;

  void DefragMemory() override;

  void SetMemPoolBlockSize(size_t available_device_mem_size) override;

  // Element in vector : memory_stream_id, address
  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                   const DeviceEventPtr &event) override;
  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) override;
  bool WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) override;
  bool SyncAllEvents() override;

  std::string GetMemoryPoolType() const override { return "Other"; }
#ifdef WITH_BACKEND

 protected:
#endif
  const MemStatusManagerPtr &common_mem() const { return common_mem_; }
  const MemStatusManagerPtr &persistent_mem() const { return persistent_mem_; }
  void *GetMinUsingMemoryAddr() const override;

  // Calculate memory block required alloc size when adding the memory block.
  size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) override;
  std::set<DeviceMemPtr> mem_bufs_;

  // The related interface of device memory eager free.
  const bool IsEnableEagerFree() const override { return false; }
  const bool IsEnableVmm() const override { return enable_vmm_; }
  void SetEnableVmm(bool enable_vmm) { enable_vmm_ = enable_vmm; }

  const size_t FreeIdleMemsByEagerFree() override;
#ifdef WITH_BACKEND

 private:
#endif
  // Find available memory buf from total pools by status, which contains idle and eager free.
  DeviceMemPtr FindAvailableMemBuf(size_t size, bool from_persistent_mem, uint32_t stream_id);
  // Find the target status memory buf from total pools by aligned size when memory alloc.
  DeviceMemPtr FindMemBufByStatus(size_t size, bool from_persistent_mem, DynamicMemBufStatus target_status,
                                  uint32_t stream_id);
  // Find the target status memory buf from specific pool by aligned size when memory alloc.
  DeviceMemPtr FindMemBufInSpecifiedMng(size_t size, bool from_persistent_mem, DynamicMemBufStatus target_status,
                                        uint32_t stream_id);

  // Add memory block and memory.
  DeviceMemPtr AddMemBlockAndMemBuf(size_t size, bool from_persistent_mem, bool need_recycle, uint32_t stream_id);
  // Add memory block and memory buf with eager free api.
  DeviceMemPtr AddMemBlockAndMemBufByEagerFree(size_t size, bool from_persistent_mem, uint32_t stream_id);
  // Add the memory block and memory buf when memory alloc not find the available memory buf.
  DeviceMemPtr CreateMemBlockAndMemBuf(size_t size, bool from_persistent_mem, DeviceMemPtr source_addr,
                                       size_t source_size, DynamicMemBufStatus mem_buf_status, uint32_t stream_id);

  // Judge whether need split the memory buf by alloc size and memory buf size.
  bool IsSplit(size_t tensor_size, size_t mem_buf_size) const;
  // Split the memory buf by alloc size.
  void SplitMemBuf(size_t size, const DynamicMemBufPtr &mem_buf, const MemStatusManagerPtr &mem_mng,
                   uint32_t stream_id);

  // Find the memory block by device address.
  DynamicMemBlockPtr FindMemBlock(const DeviceMemPtr &device_addr, const MemStatusManagerPtr &mem_mng) const;
  // The Comparator of memory block by device address, because memory blocks are arranged in order by device address.
  static bool CmpMemBlock(const DeviceMemPtr &device_addr, const DynamicMemBlockPtr &mem_block);

  // Free memory inner with no lock, the caller need lock.
  void FreeTensorMemInner(const DeviceMemPtr &device_addr);
  // Pre combine mem buf, return false when mem buf can not combine.
  bool PreCombineMemBuf(const DynamicMemBufPtr &mem_buf, const MemStatusManagerPtr &mem_mng);
  // Combine the memory buf when memory free, to avoid the memory fragmentation.
  void CombineMemBuf(const DynamicMemBlockPtr &mem_block, const DeviceAddrMapMemBuf::iterator &iter,
                     const MemStatusManagerPtr &mem_mng, DynamicMemBufStatus origin_status,
                     DynamicMemBufStatus target_status);
  // Fetch the mem info by the strict addr.
  std::tuple<DynamicMemBlockPtr, DeviceAddrMapMemBuf::iterator, MemStatusManagerPtr> FindByStrictAddr(
    const DeviceMemPtr &device_addr) const;

  // Keep the part memorys by addr.
  void KeepTensorMemByAddr(const DeviceMemPtr &device_addr, size_t size);
  std::tuple<DynamicMemBlockPtr, DynamicMemBufPtr, MemStatusManagerPtr> FindByKeepAddr(
    const DeviceMemPtr &device_addr) const;
  DynamicMemBufPtr FindMemBufByKeepAddr(const DeviceMemPtr &device_addr, const DynamicMemBlockPtr &mem_block) const;
  // Sync all events inner without lock.
  bool SyncAllEventsInner();

#ifdef __APPLE__
  // There are some problems with using mutex on Mac, use spinlocks instead.
  SpinLock spin_lock_;
#else
  // Support multi-thread.
  std::mutex mutex_;
#endif
  MemStatusManagerPtr persistent_mem_{nullptr};
  MemStatusManagerPtr common_mem_{nullptr};
  // In the graph mode, the unit size set in the context will be modified through the FetchMemUnitSize function, so it
  // needs to be changed back after that
  size_t config_unit_size_{kDynamicMemAllocUnitSize};
  // Flag for eager free routine. This flag set to false when initializing, and set to true when triggering oom.
  bool is_trigger_eager_free_{false};

  // key : <user_stream_id, memory_stream_id>
  std::unordered_map<std::pair<uint32_t, uint32_t>, std::set<DynamicMemBufPtr>, pair_hash> stream_pair_addresses_;

  bool enable_vmm_{false};
  size_t eager_free_count_{0};
  size_t last_eager_free_count_{0};
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_MEM_DYNAMIC_ALLOCATOR_H_
