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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_DYNAMIC_MEM_POOL_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_DYNAMIC_MEM_POOL_H_

#include <algorithm>
#include <functional>
#include <list>
#include <map>
#include <memory>
#include <set>
#include <unordered_map>
#include <utility>
#include <vector>
#include <string>
#include <tuple>

#include "include/backend/visible.h"
#include "include/common/utils/stream_util.h"
#include "ir/device_event.h"

namespace mindspore {
namespace device {
constexpr int kShiftOffset = 2;
constexpr int kAllocatorTypeNum = 6;
// Alloc memory aligned according to 512 bytes.
constexpr size_t kDynamicMemAlignSize = 512;
// The minimum unit size (1G) of memory block used for dynamic extend.
constexpr size_t kDynamicMemAllocUnitSize = 1024 << 20;

const char kPersistentParamMem[] = "Persistent mem";
const char kCommonMem[] = "Common mem";
constexpr size_t kMBToByte = 1024 << 10;
constexpr size_t kGBToByte = 1024 << 20;
// The smallest memory request size, if it is smaller than this size, the device memory request may fail
// Set experience value to 10M
const size_t kMinimumAllocMem = 10 << 20;

const char kBlockMemorySize[] = "block_memory_size";
const char kBlockStreamId[] = "block_stream_id";
const char kCommonMemPoolType[] = "common_mem_pool";
const char kPersistentMemPoolType[] = "persistent_mem_pool";

// The status of memory buf.
enum class DynamicMemBufStatus : int { kMemBufIdle, kMemBufUsed, kMemBufEagerFree, kMemBufUsedByEvent };
const std::string &DynamicMemBufStatusToString(DynamicMemBufStatus status);

// Memory allocator type is used to record the memory classification statistics information.
enum class AllocatorType : int { kWeight, kConstantValue, kKernelOutput, kGraphOutput, kWorkspace, kOther };
const std::string &AllocatorTypeToString(AllocatorType allocator_type);

// The Comparator of device address from small to large.
using DeviceMemPtr = void(*);
struct DeviceAddrCmp {
  bool operator()(const DeviceMemPtr &addr1, const DeviceMemPtr &addr2) const { return addr1 < addr2; }
};

// The AllocatorDebugInfo wrapper which is the local thread for the dynamic memory pool.
class DynamicMemAllocatorDebugInfo;
// Memory buf is the smallest operation object of dynamic memory pool.
struct DynamicMemBuf;
using DynamicMemBufPtr = std::shared_ptr<DynamicMemBuf>;
// Multimap key is the tensor size, for finding the idle memory buf by tensor size.
using SizeMapMemBuf = std::multimap<size_t, DynamicMemBufPtr>;
// Map key is the device address, for finding the used memory buf in memory block by device address.
using DeviceAddrMapMemBuf = std::map<DeviceMemPtr, DynamicMemBufPtr, DeviceAddrCmp>;
// Memory block is composed of memory buf.
class DynamicMemBlock;
using DynamicMemBlockPtr = std::shared_ptr<DynamicMemBlock>;

struct MemStatusManager;
using MemStatusManagerPtr = std::shared_ptr<MemStatusManager>;

// Help class for unordered_map, pair has no hash method, need override it.
struct pair_hash {
  template <class L, class R>
  std::size_t operator()(const std::pair<L, R> &param) const {
    size_t hash = std::hash<L>{}(param.first);
    hash <<= (sizeof(size_t) << kShiftOffset);
    hash ^= std::hash<R>{}(param.second);
    return std::hash<size_t>{}(hash);
  }
};

// Interface of dynamic memory pool.
class BACKEND_EXPORT DynamicMemPool {
 public:
  virtual ~DynamicMemPool() = default;

  // Release the real device memory.
  virtual void ReleaseDeviceRes() = 0;

  // The main program entry of memory alloc.
  virtual DeviceMemPtr AllocTensorMem(size_t size, bool from_persistent_mem = false, bool need_recycle = false,
                                      uint32_t stream_id = kDefaultStreamIndex) = 0;
  // The main program entry of continuous memory alloc.
  virtual std::vector<DeviceMemPtr> AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                             uint32_t stream_id = kDefaultStreamIndex) = 0;
  // The main program entry of memory free.
  virtual void FreeTensorMem(const DeviceMemPtr &device_addr) = 0;
  // The main program entry of part memorys free and part memorys keep.
  virtual void FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                  const std::vector<DeviceMemPtr> &keep_addrs,
                                  const std::vector<size_t> &keep_addr_sizes) = 0;

  // Element in vector : memory_stream_id, address
  virtual bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                           const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                           const DeviceEventPtr &event) {
    return false;
  }
  virtual bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
    return false;
  }
  virtual bool WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) { return false; }
  virtual bool SyncAllEvents() { return false; }

  // The real size by memory alloc aligned.
  virtual size_t AlignMemorySize(size_t size) const {
    if (size == 0) {
      return kDynamicMemAlignSize;
    }
    return ((size + kDynamicMemAlignSize - 1) / kDynamicMemAlignSize) * kDynamicMemAlignSize;
  }
  // Calculate memory block required alloc size when adding the memory block.
  virtual size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) {
    return kDynamicMemAllocUnitSize;
  }
  // Set mem pool block size
  virtual void SetMemPoolBlockSize(size_t available_device_mem_size) {}
  // Get the minimum memory unit size using for dynamic extend.
  virtual size_t MemAllocUnitSize(bool from_persistent_mem) const { return kDynamicMemAllocUnitSize; }
  virtual void SetMemAllocUintSize(size_t common_size, size_t persist_size = kDynamicMemAllocUnitSize) {}

  virtual void *GetMinUsingMemoryAddr() const { return nullptr; }

  // The related interface of device memory real operation, needs override by device type.
  virtual size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) = 0;
  virtual bool FreeDeviceMem(const DeviceMemPtr &addr) = 0;

  virtual size_t free_mem_size() { return 0; }
  virtual uint64_t total_mem_size() const { return 0; }
  virtual size_t GetMaxUsedMemSize() const { return 0; }
  virtual size_t GetVmmUsedMemSize() const { return 0; }

  // The related interface of device memory eager free.
  virtual void DefragMemory() {}

  // Display the brief state information of memory block and memory buf.
  virtual void DumpDynamicMemPoolStateInfo() {}
  // Display the detailed debug information of memory block and memory buf.
  virtual void DumpDynamicMemPoolDebugInfo() {}

  // The statistics information.
  virtual size_t TotalMemStatistics() const = 0;
  virtual size_t TotalUsedMemStatistics() const = 0;
  virtual size_t TotalUsedByEventMemStatistics() const = 0;
  virtual size_t TotalIdleMemStatistics() const = 0;
  virtual size_t TotalEagerFreeMemStatistics() const = 0;
  virtual size_t UsedMemPeakStatistics() const = 0;
  virtual size_t MaxMemAllocatedStatistics() const = 0;
  virtual size_t MaxMemReservedStatistics() const = 0;
  virtual size_t ActualPeakStatistics() const = 0;
  virtual std::unordered_map<std::string, std::size_t> BlockCountsStatistics() const = 0;
  virtual std::unordered_map<std::string, std::size_t> BlockUnitSizeStatistics() const = 0;
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  CommonMemBlocksInfoStatistics() const = 0;
  virtual std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
  PersistentMemBlocksInfoStatistics() const = 0;
  virtual void ResetMaxMemReserved() = 0;
  virtual void ResetMaxMemAllocated() = 0;

  virtual std::string GetMemoryPoolType() const { return "Other"; }

 protected:
  virtual const bool IsEnableEagerFree() const { return false; }
  virtual const bool IsEnableVmm() const { return false; }
  virtual const bool SyncAllStreams() { return false; }
  virtual size_t AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr *addr) { return 0; }
  virtual size_t FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) { return 0; }
  virtual size_t MmapDeviceMem(size_t size, DeviceMemPtr addr) { return 0; }
  virtual const size_t FreeIdleMemsByEagerFree() { return 0; }
};

// Recording information for debugging the memory allocator.
struct AllocatorDebugInfo {
  std::string name_{"Unknown"};
  AllocatorType type_{AllocatorType::kOther};
  int input_index_{-1};
  int output_index_{-1};
};

class DynamicMemAllocatorDebugInfo {
 public:
  static AllocatorDebugInfo &GetDebugInfo() noexcept { return debug_info_; }

  // Set the debug info when memory alloc.
  static void SetDebugInfo(const std::string &name, AllocatorType type, int input_index = -1, int output_index = -1) {
    debug_info_.name_ = name;
    debug_info_.type_ = type;
    debug_info_.input_index_ = input_index;
    debug_info_.output_index_ = output_index;
  }

 private:
  DynamicMemAllocatorDebugInfo() = default;
  virtual ~DynamicMemAllocatorDebugInfo() = default;
  DynamicMemAllocatorDebugInfo(const DynamicMemAllocatorDebugInfo &) = delete;
  DynamicMemAllocatorDebugInfo &operator=(const DynamicMemAllocatorDebugInfo &) = delete;

  static thread_local AllocatorDebugInfo debug_info_;
};

using TaskIdOnStreamEvent = std::pair<int64_t, DeviceEventPtr>;
struct EventBase {
  // Record event on mem buf.
  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id, const DeviceEventPtr &event);

  // Release events on mem buf.
  bool WaitEvent(uint32_t task_id_on_stream, uint32_t user_stream_id);

  // Indicates if mem buf used by event, return true when no event bind on mem buf.
  bool IsEventNotUsed();

  // Sync all events that bound on mem buf.
  bool SyncAllEvents();

  // Parameter: user_stream_id, list of <task_id_on_stream, event>.
  std::shared_ptr<std::unordered_map<uint32_t, std::shared_ptr<std::list<TaskIdOnStreamEvent>>>> events_{nullptr};
};

struct DynamicMemBuf : public EventBase {
  DynamicMemBuf(DeviceMemPtr addr, DynamicMemBufStatus status, size_t size, uint32_t stream_id)
      : device_addr_(addr), status_(status), size_(size), stream_id_(stream_id) {}
  DynamicMemBuf(DeviceMemPtr addr, DynamicMemBufStatus status, size_t size, uint32_t stream_id,
                const std::string &allocator_name, AllocatorType allocator_type)
      : device_addr_(addr),
        status_(status),
        size_(size),
        stream_id_(stream_id),
        allocator_name_(allocator_name),
        allocator_type_{allocator_type} {}
  DynamicMemBuf(const DynamicMemBuf &) = delete;
  DynamicMemBuf &operator=(const DynamicMemBuf &) = delete;

  DeviceMemPtr device_addr_;
  DynamicMemBufStatus status_;
  size_t size_;

  uint32_t stream_id_{0};

  // Debug info.
  std::string allocator_name_;
  AllocatorType allocator_type_{AllocatorType::kOther};
};

class DynamicMemBlock {
 public:
  DynamicMemBlock(DeviceMemPtr addr_base, size_t size, const uint32_t stream_id)
      : device_addr_base_(addr_base), mem_block_size_(size), stream_id_(stream_id) {}
  DynamicMemBlock() = delete;
  DynamicMemBlock(const DynamicMemBlock &) = delete;
  DynamicMemBlock &operator=(const DynamicMemBlock &) = delete;

  ~DynamicMemBlock() { block_all_mem_buf_map_.clear(); }

  const DeviceMemPtr &device_addr() const { return device_addr_base_; }

  size_t size() const { return mem_block_size_; }

  void update_border_addr(DeviceMemPtr left_addr, DeviceMemPtr right_addr);

  size_t get_actual_peak();

  // The map of all memory buf in this memory block by device address.
  DeviceAddrMapMemBuf block_all_mem_buf_map_;

  DeviceMemPtr device_addr_base_{nullptr};

  // Max addr
  DeviceMemPtr max_addr_ = nullptr;
  // Min addr
  DeviceMemPtr min_addr_ = nullptr;

  size_t mem_block_size_{0};
  const uint32_t stream_id_;
};

struct DeviceState {
  void UpdatePeakSize() {
    size_t total_used_size_ = total_used_mem_size_ + total_used_by_event_mem_size_;
    size_t temp_used_size_ = temp_total_used_mem_size_ + temp_total_used_by_event_mem_size_;
    used_mem_peak_size_ = std::max(used_mem_peak_size_, total_used_size_);
    if (total_used_size_ > temp_used_size_) {
      temp_used_mem_peak_size_ = std::max(temp_used_mem_peak_size_, total_used_size_ - temp_used_size_);
    }
  }

  // Memory allocated from device
  size_t total_mem_size_{0};
  // Memory in use
  size_t total_used_mem_size_{0};
  // Memory in use by event
  size_t total_used_by_event_mem_size_{0};
  // Memory in idle.
  size_t total_idle_mem_size_{0};
  // Memory in eager free.
  size_t total_eager_free_mem_size_{0};
  // Maximum peak memory usage
  size_t used_mem_peak_size_{0};
  // Recorded data for memory in use since reset maximum allocated memory
  size_t temp_total_used_mem_size_{0};
  // Recorded data for memory in use by event since reset maximum allocated memory
  size_t temp_total_used_by_event_mem_size_{0};
  // Recorded data for maximum peak memory usage since reset maximum allocated memory
  size_t temp_used_mem_peak_size_{0};
  // Temporary recorded data for memory reserved since reset maximum reserved memory
  size_t temp_total_mem_size_{0};
};

struct MemStatusManager {
  bool Empty() const { return mem_block_list_.empty(); }

  void AddMemBlock(const DynamicMemBlockPtr &mem_block, uint32_t stream_id);

  void DoAddMemBlock(const DynamicMemBlockPtr &mem_block, std::vector<DynamicMemBlockPtr> *mem_block_list);

  size_t CalActualPeak();

  SizeMapMemBuf &GetOrCreateMemBufMap(uint32_t stream_id, DynamicMemBufStatus status);

  void AddMemBuf(const DynamicMemBufPtr &mem_buf);

  void RemoveMemBuf(const DynamicMemBufPtr &mem_buf);

  void Clear() noexcept;

  const DeviceState DumpMemBlockDebugInfo(const std::string &mem_type);

  std::vector<uint32_t> GetStreamIds() const {
    std::vector<uint32_t> stream_ids;
    for (const auto &iter : mem_blocks_) {
      (void)stream_ids.emplace_back(iter.first);
    }
    return stream_ids;
  }

  size_t unit_size_{kDynamicMemAllocUnitSize};
  // Mem pool state
  DeviceState mps_;

  std::vector<DynamicMemBlockPtr> mem_block_list_;
  std::vector<DynamicMemBlockPtr> mem_block_insertion_order_;
  size_t total_block_size_ = 0;
  std::unordered_map<uint32_t, std::vector<DynamicMemBlockPtr>> mem_blocks_;
  std::unordered_map<std::pair<uint32_t, DynamicMemBufStatus>, SizeMapMemBuf, pair_hash> mem_bufs_;
};
}  // namespace device
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_MEM_REUSE_DYNAMIC_MEM_POOL_H_
