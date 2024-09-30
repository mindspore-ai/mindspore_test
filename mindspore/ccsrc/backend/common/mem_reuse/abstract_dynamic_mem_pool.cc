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

#include "include/backend/mem_reuse/abstract_dynamic_mem_pool.h"

#include <stdio.h>

#include <algorithm>
#include <atomic>

#include "include/backend/distributed/collective/collective_manager.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "include/common/debug/common.h"
#include "include/common/utils/comm_manager.h"
#include "include/common/utils/utils.h"
#include "utils/convert_utils_base.h"
#include "utils/file_utils.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {

LockGuard::LockGuard(const Lock &lock) : lock_(const_cast<Lock *>(&lock)) { lock_->lock(); }

LockGuard::~LockGuard() { lock_->unlock(); }

MemBuf::MemBuf(size_t size, void *addr, uint32_t stream_id, MemBlock *mem_block, MemBufStatus status)
    : prev_(nullptr),
      next_(nullptr),
      size_(size),
      addr_(addr),
      stream_id_(stream_id),
      mem_block_(mem_block),
      status_(status) {}

MemBuf::~MemBuf() {}

MemBufAllocator::~MemBufAllocator() {
  MS_LOG(INFO) << "MemBufAllocator[" << this << "] : " << BriefInfo() << " deconstruct.";
  for (auto &mem_block : mem_blocks_) {
    delete mem_block;
  }
  mem_blocks_.clear();
  for (auto mem_buf : free_mem_bufs_) {
    delete mem_buf;
  }
  free_mem_bufs_.clear();
  for (auto mem_buf : eager_free_mem_bufs_) {
    delete mem_buf;
  }
  eager_free_mem_bufs_.clear();
  delete search_key_;
}

void MemBufAllocator::ReleaseDeviceRes() {
  MS_LOG(INFO) << "Release device resource for allocator, " << BriefInfo()
               << ", mem_blocks_ size : " << mem_blocks_.size() << ".";
  for (auto mem_block : mem_blocks_) {
    MS_LOG(INFO) << "Clean mem block : " << mem_block->ToJson() << ".";
    (void)mem_block_cleaner_(mem_block);
  }
  for (auto mem_block : mem_blocks_) {
    MS_LOG(INFO) << "Delete mem block : " << mem_block->ToJson() << ".";
    delete mem_block;
  }
  mem_blocks_.clear();

  MS_LOG(INFO) << "Free mem buf size : " << free_mem_bufs_.size() << ".";
  for (auto mem_buf : free_mem_bufs_) {
    delete mem_buf;
  }
  free_mem_bufs_.clear();

  MS_LOG(INFO) << "Eager free mem buf size : " << eager_free_mem_bufs_.size() << ".";
  for (auto mem_buf : eager_free_mem_bufs_) {
    delete mem_buf;
  }
  eager_free_mem_bufs_.clear();
}

MemBuf *MemBufAllocator::Malloc(size_t size) {
  // Malloc with expand block first.
  if (MS_UNLIKELY(mem_blocks_.empty())) {
    return MallocExpandBlock(size);
  }

  search_key_->size_ = size;
  auto it = free_mem_bufs_.lower_bound(search_key_);
  MemBuf *candidate = nullptr;
  if (MS_LIKELY(it != free_mem_bufs_.end())) {
    candidate = *it;
    (void)free_mem_bufs_.erase(it);
  } else {
    it = eager_free_mem_bufs_.lower_bound(search_key_);
    if (it != eager_free_mem_bufs_.end()) {
      candidate = *it;
      (void)eager_free_mem_bufs_.erase(it);
    }
  }
  if (MS_UNLIKELY(candidate == nullptr)) {
    return nullptr;
  }

  return MapAndSplitMemBuf(candidate, size);
}

bool MemBufAllocator::Free(MemBuf *mem_buf, MemBufStatus target_status) {
  // Change mem buf status to used by event, and wait for event to free.
  if (MS_UNLIKELY(!mem_buf->IsEventNotUsed())) {
    mem_buf->status_ = MemBufStatus::kMemBufUsedByEvent;
    return false;
  }

  mem_buf->status_ = target_status;
  // Try to merge from prev.
  auto prev_buf = mem_buf->prev_;
  if (MS_LIKELY(prev_buf != nullptr && prev_buf->status_ == target_status)) {
    // Erase prev buf pointer
    auto prev = prev_buf->prev_;
    mem_buf->prev_ = prev;
    if (prev != nullptr) {
      prev->next_ = mem_buf;
    }

    mem_buf->addr_ = prev_buf->addr_;
    mem_buf->size_ += prev_buf->size_;
    if (target_status == MemBufStatus::kMemBufIdle) {
      auto ret = free_mem_bufs_.erase(prev_buf);
      if (ret == 0) {
        MS_LOG(ERROR) << "Erase mem buf : " << mem_buf->ToJson() << " prev buf " << prev_buf->ToJson() << " failed.";
      }
    } else if (target_status == MemBufStatus::kMemBufEagerFree) {
      auto ret = eager_free_mem_bufs_.erase(prev_buf);
      if (ret == 0) {
        MS_LOG(ERROR) << "Erase mem buf : " << mem_buf->ToJson() << " prev buf " << prev_buf->ToJson() << " failed.";
      }
    }
    delete prev_buf;
  }
  // Try to merge from next.
  auto next_buf = mem_buf->next_;
  if (MS_LIKELY(next_buf != nullptr && next_buf->status_ == target_status)) {
    // Erase next buf pointer
    auto next = next_buf->next_;
    mem_buf->next_ = next;
    if (next != nullptr) {
      next->prev_ = mem_buf;
    }

    mem_buf->size_ += next_buf->size_;
    if (target_status == MemBufStatus::kMemBufIdle) {
      auto ret = free_mem_bufs_.erase(next_buf);
      if (ret == 0) {
        MS_LOG(ERROR) << "Erase next buf : " << next_buf->ToJson() << " failed.";
      }
    } else if (target_status == MemBufStatus::kMemBufEagerFree) {
      auto ret = eager_free_mem_bufs_.erase(next_buf);
      if (ret == 0) {
        MS_LOG(ERROR) << "Erase next buf : " << next_buf->ToJson() << " failed.";
      }
    }
    delete next_buf;
  }

  if (target_status == MemBufStatus::kMemBufIdle) {
    (void)free_mem_bufs_.emplace(mem_buf);
  } else if (target_status == MemBufStatus::kMemBufEagerFree) {
    (void)eager_free_mem_bufs_.emplace(mem_buf);
  }

  return true;
}

MemBuf *MemBufAllocator::MallocExpandBlock(size_t size) {
  MemBlock *mem_block = ExpandBlock(size);
  if (mem_block == nullptr) {
    return nullptr;
  }
  MemBuf *candidate = new MemBuf(mem_block->size_, mem_block->addr_, mem_block->stream_id_, mem_block,
                                 enable_eager_free_ ? MemBufStatus::kMemBufEagerFree : MemBufStatus::kMemBufIdle);
  if (candidate->size_ < size) {
    if (candidate->status_ == MemBufStatus::kMemBufIdle) {
      (void)free_mem_bufs_.emplace(candidate);
    } else {
      (void)eager_free_mem_bufs_.emplace(candidate);
    }
    MS_LOG(ERROR) << "Candidate size: " << candidate->size_ << " is less than required size : " << size << ".";
    return nullptr;
  }

  return MapAndSplitMemBuf(candidate, size);
}

const std::pair<size_t, size_t> MemBufAllocator::FreeIdleMemsByEagerFree() {
  // Free all idle mem bufs.
  size_t eager_free_size = 0;
  for (auto mem_buf : free_mem_bufs_) {
    eager_free_size += mem_buf->size_;
    Free(mem_buf, MemBufStatus::kMemBufEagerFree);
  }
  free_mem_bufs_.clear();
  // Do eager free on eager free mem bufs.
  size_t real_free_size = 0;
  for (auto mem_buf : eager_free_mem_bufs_) {
    MS_LOG(DEBUG) << "Eager free mem buf : " << mem_buf << ", details : " << mem_buf->ToJson() << ".";
    real_free_size += mem_eager_freer_(mem_buf->addr_, mem_buf->size_);
  }
  MS_LOG(INFO) << "Free idle mems by eager free, egaer_free_size : " << eager_free_size
               << ", real_free_size : " << real_free_size << ".";
  return std::make_pair(eager_free_size, real_free_size);
}

std::string MemBufAllocator::DumpStateInfo() const {
  std::stringstream ss;
  ss << "Dump state info for " << BriefInfo() << "\n";
  ss << "\tMem block info - total count : " << mem_blocks_.size() << "\n";
  size_t mem_block_size = 0;
  for (const auto mem_block : mem_blocks_) {
    ss << "\t\t" << mem_block->ToJson() << "\n";
    mem_block_size += mem_block->size_;
  }
  ss << "\tMem block total size : " << mem_block_size << "\n";

  auto mem_buf_accumulter = [](size_t size, MemBuf *const &left) { return left->size_ + size; };
  size_t init_value = 0;
  size_t free_size = std::accumulate(free_mem_bufs_.begin(), free_mem_bufs_.end(), init_value, mem_buf_accumulter);
  ss << "\tFree mem buf total size : " << free_size << ", total count : " << free_mem_bufs_.size() << "\n";

  size_t eager_free_size =
    std::accumulate(eager_free_mem_bufs_.begin(), eager_free_mem_bufs_.end(), init_value, mem_buf_accumulter);
  ss << "\tEager free mem buf total size : " << eager_free_size << ", total count : " << eager_free_mem_bufs_.size()
     << "\n";

  return ss.str();
}

std::string MemBufAllocator::DumpDebugInfo() const {
  std::stringstream ss;
  ss << "Dump debug info for " << BriefInfo() << "\n";
  ss << "\tMem block info - total count : " << mem_blocks_.size() << "\n";
  for (const auto mem_block : mem_blocks_) {
    ss << "\t\t" << mem_block->ToJson() << "\n";
  }
  ss << "\tFree mem buf info - total count : " << free_mem_bufs_.size() << "\n";
  for (const auto mem_buf : free_mem_bufs_) {
    ss << "\t\t" << mem_buf->ToJson() << "\n";
  }
  ss << "\tEager free mem buf info - total count : " << eager_free_mem_bufs_.size() << "\n";
  for (const auto mem_buf : eager_free_mem_bufs_) {
    ss << "\t\t" << mem_buf->ToJson() << "\n";
  }
  return ss.str();
}

MemBuf *MemBufAllocator::MapAndSplitMemBuf(MemBuf *candidate, size_t size) {
  size_t remaining_size = candidate->size_ - size;
  // Mmap memory first.
  if (candidate->status_ == MemBufStatus::kMemBufEagerFree) {
    size_t map_size = (remaining_size >= kDynamicMemAlignSize) ? size : candidate->size_;
    auto mapped_size = mem_mapper_(map_size, candidate->addr_);
    if (mapped_size != map_size) {
      MS_LOG(INFO) << "Mapped_size : " << mapped_size << " is not equal to required size : " << map_size
                   << ", mem buf info : " << candidate->ToJson() << ".";
      (void)eager_free_mem_bufs_.emplace(candidate);
      return nullptr;
    }
  }

  // Try to split mem buf.
  if (MS_LIKELY(remaining_size >= kDynamicMemAlignSize)) {
    void *remaining_addr = static_cast<uint8_t *>(candidate->addr_) + size;
    auto remaining_buf =
      new MemBuf(remaining_size, remaining_addr, candidate->stream_id_, candidate->mem_block_, candidate->status_);

    auto next = candidate->next_;
    if (next != nullptr) {
      next->prev_ = remaining_buf;
      remaining_buf->next_ = next;
    }
    candidate->next_ = remaining_buf;
    remaining_buf->prev_ = candidate;
    if (remaining_buf->status_ == MemBufStatus::kMemBufIdle) {
      (void)free_mem_bufs_.emplace(remaining_buf);
    } else {
      (void)eager_free_mem_bufs_.emplace(remaining_buf);
    }

    // Update candidate size.
    candidate->size_ = size;
  }

  candidate->status_ = MemBufStatus::kMemBufUsed;
  // Update mem block usage.
  candidate->mem_block_->UpdateBorderAddr(candidate);

  return candidate;
}

MemBlock *MemBufAllocator::ExpandBlock(size_t size) {
  MemBlock *mem_block = mem_block_expander_(size);
  if (mem_block == nullptr) {
    MS_LOG(ERROR) << "Expand block failed, expand size : 0.";
    return nullptr;
  }

  if (mem_block->size_ < size) {
    MS_LOG(ERROR) << "Expand block failed, expand size : " << mem_block->size_ << ".";
  }

  (void)mem_blocks_.emplace_back(mem_block);
  return mem_block;
}

AbstractDynamicMemPool::AbstractDynamicMemPool() {}

void AbstractDynamicMemPool::ReleaseDeviceRes() {
  LockGuard lock(lock_);
  DumpDynamicMemPoolStateInfo();
  for (const auto &iter : stream_pair_mem_bufs_) {
    auto size = iter.second.size();
    MS_LOG(INFO) << "Event referred stream_pair_mem_bufs_[" << iter.first.first << "-" << iter.first.second
                 << "], size : " << size << ".";
  }
  // Clear map of address to mem buf.
  for (const auto &iter : addr_mem_buf_allocators_) {
    auto mem_buf = iter.second.first;
    delete mem_buf;
  }
  addr_mem_buf_allocators_.clear();

  MS_LOG(INFO) << "Release device resource for " << GetMemoryPoolType() << " : " << mem_stat_.ToReadableString() << ".";
  for (const auto &stream_id_allocator : stream_id_allocators_) {
    const auto &allocator = stream_id_allocator.second;
    allocator->ReleaseDeviceRes();
  }
  stream_id_allocators_.clear();
  stream_pair_mem_bufs_.clear();
  mem_stat_.Reset();
}

// Allocation follow steps below:
//  1 align size
//  2 find from current allocator, if failed transfer to 3
//  3 find from another allocator, if failed transfer to 4
//  4 do eager free and find from current allocator again, if failed transfer to 5
//  5 expand block
DeviceMemPtr AbstractDynamicMemPool::AllocTensorMem(size_t size, bool from_persistent_mem, bool, uint32_t stream_id) {
  size_t align_size = AlignMemorySize(size);
  LockGuard lock(lock_);
  auto &&mem_buf_allocator = AllocMemBuf(align_size, from_persistent_mem, stream_id);
  if (MS_UNLIKELY(mem_buf_allocator.first == nullptr)) {
    // Dump mem pool state info when alloc tensor failed.
    DumpDynamicMemPoolStateInfo();
    return nullptr;
  }

  (void)addr_mem_buf_allocators_.emplace(mem_buf_allocator.first->addr_, mem_buf_allocator);
  return mem_buf_allocator.first->addr_;
}

std::pair<MemBuf *, MemBufAllocator *> AbstractDynamicMemPool::AllocMemBuf(size_t align_size, bool from_persistent_mem,
                                                                           uint32_t stream_id) {
  auto allocator = GetMemBufAllocator(align_size, from_persistent_mem, stream_id);
  auto mem_buf = allocator->Malloc(align_size);
  if (MS_UNLIKELY(mem_buf == nullptr)) {
    // Enable malloc from another allocator when from_persisatent_mem is true and vmm is not enabled.
    if (!enable_vmm_ && from_persistent_mem) {
      auto another_allocator = GetMemBufAllocator(align_size, !from_persistent_mem, stream_id);
      mem_buf = another_allocator->Malloc(align_size);
      if (MS_UNLIKELY(mem_buf != nullptr)) {
        allocator = another_allocator;
      }
    }

    if (MS_UNLIKELY(mem_buf == nullptr)) {
      if (IsEnableEagerFree() || enable_vmm_) {
        if (!SyncAllStreams()) {
          MS_LOG(INTERNAL_EXCEPTION) << "Sync all streams failed.";
          return std::make_pair(nullptr, nullptr);
        }
        (void)FreeIdleMemsByEagerFree();
        mem_buf = allocator->Malloc(align_size);
      }
      if (MS_UNLIKELY(mem_buf == nullptr)) {
        mem_buf = allocator->MallocExpandBlock(align_size);
        if (MS_UNLIKELY(mem_buf == nullptr)) {
          MS_LOG(INFO) << "Alloc tensor mem failed and try to sync all events to release memory.";
          (void)DoSyncAllEvents();
          mem_buf = allocator->Malloc(align_size);
          if (MS_UNLIKELY(mem_buf == nullptr)) {
            return std::make_pair(nullptr, nullptr);
          }
        }
      }
    }
  }

  // Update stat.
  mem_stat_.used_size_ += mem_buf->size_;
  mem_stat_.UpdatePeakSize();
  return std::make_pair(mem_buf, allocator);
}

std::vector<DeviceMemPtr> AbstractDynamicMemPool::AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                                           uint32_t stream_id) {
  std::vector<DeviceMemPtr> device_addr_list;
  size_t total_size = std::accumulate(size_list.begin(), size_list.end(), IntToSize(0));
  // Pre-alloc the one whole piece memory.
  auto device_addr = AllocTensorMem(total_size, false, false, stream_id);
  if (!device_addr) {
    return device_addr_list;
  }

  (void)device_addr_list.emplace_back(device_addr);
  if (size_list.size() == 1) {
    return device_addr_list;
  }

  // Try to split mem bufs.
  LockGuard lock(lock_);
  auto &&it = addr_mem_buf_allocators_.find(device_addr);
  if (it != addr_mem_buf_allocators_.end()) {
    auto mem_buf = it->second.first;
    auto allocator = it->second.second;
    mem_buf->size_ = size_list[0];
    MemBuf *prev_mem_buf = mem_buf;
    void *next_addr = static_cast<uint8_t *>(mem_buf->addr_) + size_list[0];
    total_size -= size_list[0];
    for (size_t i = 1; i < size_list.size(); i++) {
      auto new_mem_buf = new MemBuf(size_list[i], next_addr, stream_id, mem_buf->mem_block_, MemBufStatus::kMemBufUsed);
      new_mem_buf->Link(prev_mem_buf, prev_mem_buf->next_);
      (void)addr_mem_buf_allocators_.emplace(new_mem_buf->addr_, std::make_pair(new_mem_buf, allocator));
      // Update result.
      (void)device_addr_list.emplace_back(next_addr);
      // Update next addr and prev mem buf.
      if (i < size_list.size() - 1) {
        next_addr = static_cast<uint8_t *>(next_addr) + size_list[i];
        total_size -= size_list[i];
        prev_mem_buf = new_mem_buf;
      } else {
        // Update last mem buf
        if (total_size != size_list[i]) {
          MS_LOG(WARNING) << "Remain size : " << total_size << " is not equal to last size : " << size_list[i] << ".";
          new_mem_buf->size_ = total_size;
        }
      }
    }
  } else {
    DumpDynamicMemPoolDebugInfo();
    // Unreachable routine.
    MS_LOG(INTERNAL_EXCEPTION) << "Find addr : " << device_addr << " failed.";
  }

  return device_addr_list;
}

// The main program entry of memory free.
void AbstractDynamicMemPool::FreeTensorMem(const DeviceMemPtr &device_addr) {
  LockGuard lock(lock_);
  (void)DoFreeTensorMem(device_addr);
}

// The main program entry of memory free.
bool AbstractDynamicMemPool::DoFreeTensorMem(const DeviceMemPtr &device_addr) {
  void *addr = device_addr;
  auto &&it = addr_mem_buf_allocators_.find(device_addr);
  if (MS_LIKELY(it != addr_mem_buf_allocators_.end())) {
    auto allocator = it->second.second;
    auto mem_buf = it->second.first;
    auto free_size = mem_buf->size_;
    if (MS_LIKELY(allocator->Free(mem_buf))) {
      mem_stat_.used_size_ -= free_size;
      (void)addr_mem_buf_allocators_.erase(it);
      return true;
    }
  } else {
    // This may be normal case.
    MS_LOG(INFO) << "Free tensor mem failed, can not find address : " << addr << ".";
  }
  return false;
}

MemBufAllocator *AbstractDynamicMemPool::GetMemBufAllocator(size_t size, bool from_persistent_mem, uint32_t stream_id) {
  auto key = std::make_pair(from_persistent_mem, stream_id);
  MemBufAllocatorPtr allocator = nullptr;
  auto &&it = stream_id_allocators_.find(key);
  if (it == stream_id_allocators_.end()) {
    allocator = GenerateAllocator(from_persistent_mem, stream_id);
    (void)stream_id_allocators_.emplace(key, allocator);
  } else {
    allocator = it->second;
  }
  return allocator.get();
}

// keep addrs is in free addrs, so here find mem bufs first
// and then, traverse keep addrs and spilt candidates
void AbstractDynamicMemPool::FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                const std::vector<DeviceMemPtr> &keep_addrs,
                                                const std::vector<size_t> &keep_addr_sizes) {
  MS_LOG(DEBUG) << "Free part tensor mems.";
  LockGuard lock(lock_);
  (void)DoFreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
}

std::vector<MemBuf *> AbstractDynamicMemPool::DoFreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                                   const std::vector<DeviceMemPtr> &keep_addrs,
                                                                   const std::vector<size_t> &keep_addr_sizes) {
  std::vector<MemBuf *> mem_bufs;
  std::map<void *, std::pair<MemBuf *, MemBufAllocator *>> candidates;
  for (const auto &free_addr : free_addrs) {
    auto &&it = addr_mem_buf_allocators_.find(free_addr);
    if (it != addr_mem_buf_allocators_.end()) {
      (void)candidates.emplace(it->first, it->second);
    } else {
      // This is illegal routine, but level0 case entered.
      MS_LOG(INFO) << "Find address : " << free_addr << " failed.";
    }
  }

  std::set<size_t> processed_keep_addrs;
  for (size_t i = 0; i < keep_addrs.size(); i++) {
    auto keep_addr = keep_addrs[i];
    if (processed_keep_addrs.count((size_t)keep_addr) > 0) {
      MS_LOG(INFO) << "Duplicate keep address : " << keep_addr << ".";
      continue;
    }
    (void)processed_keep_addrs.insert((size_t)keep_addr);
    auto &&it = candidates.upper_bound(keep_addr);
    if (it == candidates.begin()) {
      MS_LOG(WARNING) << "Locate keep addr : " << keep_addr << " failed.";
      continue;
    }
    auto iter = --it;
    auto mem_buf = iter->second.first;
    auto allocator = iter->second.second;
    size_t base_start = (size_t)mem_buf->addr_;
    size_t base_end = base_start + mem_buf->size_;
    size_t keep_start = (size_t)keep_addr;
    size_t keep_end = keep_start + (size_t)keep_addr_sizes[i];
    // Since free part tensor mem may double free keep addr, continue for these keep addrs.
    if (keep_start >= base_end) {
      MS_LOG(WARNING) << "Check range error, base start : " << base_start << ", base end : " << base_end
                      << ", keep start : " << keep_start << ", keep end : " << keep_end << ".";
      continue;
    }
    // Split candidates. If keep start euqal to base start, split mem buf into two parts, or three parts.
    // First construct keep mem buf and set it into addr_mem_buf_allocators_, then process head buf and tail buf.
    MemBuf *keep_mem_buf = nullptr;
    if (keep_start == base_start) {
      keep_mem_buf = mem_buf;
      keep_mem_buf->size_ = keep_addr_sizes[i];
      // Remove keep addr since keep start equal to base start, no need to free keep addr any more.
      (void)candidates.erase(mem_buf->addr_);
    } else {
      // Split middle mem buf.
      keep_mem_buf =
        new MemBuf(keep_addr_sizes[i], keep_addr, mem_buf->stream_id_, mem_buf->mem_block_, mem_buf->status_);
      keep_mem_buf->Link(mem_buf, mem_buf->next_);
      (void)addr_mem_buf_allocators_.emplace(keep_addr, std::make_pair(keep_mem_buf, allocator));
      size_t prev_remain_size = keep_start - base_start;
      mem_buf->size_ = prev_remain_size;
    }
    (void)mem_bufs.emplace_back(keep_mem_buf);
    MS_LOG(DEBUG) << "keep_mem_buf : " << keep_mem_buf->ToJson() << ".";
    // Process last mem buf.
    if (keep_end < base_end) {
      void *last_addr = static_cast<uint8_t *>(keep_mem_buf->addr_) + keep_mem_buf->size_;
      auto last_mem_buf = new MemBuf(base_end - keep_end, last_addr, keep_mem_buf->stream_id_, keep_mem_buf->mem_block_,
                                     mem_buf->status_);
      last_mem_buf->Link(keep_mem_buf, keep_mem_buf->next_);
      (void)addr_mem_buf_allocators_.emplace(last_mem_buf->addr_, std::make_pair(last_mem_buf, allocator));
      if (candidates.count(last_mem_buf->addr_) > 0) {
        MS_LOG(INFO) << "Duplicate address : " << last_mem_buf->addr_ << ".";
      }
      MS_LOG(DEBUG) << "last mem buf : " << last_mem_buf->ToJson() << ".";
      (void)candidates.emplace(last_mem_buf->addr_, std::make_pair(last_mem_buf, allocator));
    }
  }
  for (const auto &candidate : candidates) {
    auto mem_buf = candidate.second.first;
    if (!AbstractDynamicMemPool::DoFreeTensorMem(mem_buf->addr_)) {
      DumpDynamicMemPoolStateInfo();
      MS_LOG(ERROR) << "Free device address failed : " << mem_buf->addr_ << ", mem_buf : " << mem_buf->ToJson() << ".";
    }
  }
  return mem_bufs;
}

MemBufAllocatorPtr AbstractDynamicMemPool::GenerateAllocator(bool is_persistent, uint32_t stream_id) {
  MS_LOG(INFO) << "Generate allocator, is persistent : " << is_persistent << ", stream id : " << stream_id << ".";
  std::function<MemBlock *(size_t)> mem_block_expander = [&, is_persistent = is_persistent,
                                                          stream_id = stream_id](size_t size) {
    size_t block_size = CalMemBlockAllocSize(size, is_persistent);
    MemBlock *mem_block = nullptr;
    if (block_size == 0) {
      MS_LOG(INFO) << "Malloc mem block failed, is enable eager free : " << IsEnableEagerFree()
                   << ", is enable vmm : " << IsEnableVmm() << ", size : " << size << ", block size is  0.";
      return mem_block;
    }
    DeviceMemPtr addr = nullptr;
    size_t alloc_size;
    MS_LOG(INFO) << "Malloc mem block, is enable eager free : " << IsEnableEagerFree()
                 << ", is enable vmm : " << IsEnableVmm() << ", size : " << size << ", block size : " << block_size
                 << ".";
    if (IsEnableEagerFree() || IsEnableVmm()) {
      // Virtual address is unlimited.
      auto eager_free_size = std::max(block_size, static_cast<size_t>(total_mem_size()));
      alloc_size = AllocDeviceMemByEagerFree(eager_free_size, &addr);
      mem_stat_.eager_free_size_ += alloc_size;
    } else {
      alloc_size = AllocDeviceMem(block_size, &addr);
      if (alloc_size < block_size) {
        MS_LOG(WARNING) << "Alloc device mem failed, alloc size : " << alloc_size << ", block size : " << block_size
                        << ".";
      }
    }
    if (alloc_size == 0) {
      return mem_block;
    }
    mem_stat_.alloc_size_ += alloc_size;
    mem_block = new MemBlock(alloc_size, addr, stream_id);
    MS_LOG(INFO) << "Malloc mem block : " << mem_block->ToJson() << ".";
    return mem_block;
  };

  std::function<bool(MemBlock *)> mem_block_cleaner = [&](MemBlock *mem_block) {
    mem_stat_.alloc_size_ -= mem_block->size_;
    // Call free device mem as ascend memory pool would do stat in free operation.
    return FreeDeviceMem(mem_block->addr_);
  };
  std::function<size_t(size_t size, void *addr)> mem_mapper = [&](size_t size, void *addr) {
    mem_stat_.eager_free_size_ -= size;
    return MmapDeviceMem(size, addr);
  };
  std::function<size_t(void *addr, const size_t size)> mem_eager_freer = [&](void *addr, const size_t size) {
    MS_LOG(DEBUG) << "Eager free addr : " << addr << ", size : " << size << ".";
    return FreeDeviceMemByEagerFree(addr, size);
  };

  return std::make_shared<MemBufAllocator>(mem_block_expander, mem_block_cleaner, mem_mapper, mem_eager_freer,
                                           IsEnableEagerFree() || IsEnableVmm(), is_persistent, stream_id);
}

// Element in vector : memory_stream_id, addr
bool AbstractDynamicMemPool::RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                                         const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                                         const DeviceEventPtr &event) {
  MS_LOG(DEBUG) << "Record event for task id on stream : " << task_id_on_stream
                << ", user stream id : " << user_stream_id << ".";
  LockGuard lock(lock_);
  for (auto &[memory_stream_id, addr] : memory_stream_addresses) {
    auto &&it = addr_mem_buf_allocators_.find(addr);
    if (it != addr_mem_buf_allocators_.end()) {
      auto mem_buf = it->second.first;
      if (mem_buf->IsEventNotUsed()) {
        mem_stat_.used_by_event_size_ += mem_buf->size_;
      }
      MS_LOG(DEBUG) << "Record event for : " << mem_buf->ToJson() << ".";
      (void)mem_buf->RecordEvent(task_id_on_stream, user_stream_id, event);
      (void)stream_pair_mem_bufs_[std::make_pair(user_stream_id, memory_stream_id)].emplace(mem_buf);
    } else {
      // Output of somas sub graph may be used by somas sub graph inner node, address may not be kept in mem pool.
      MS_LOG(DEBUG) << "Unknown address : " << addr << ".";
    }
  }
  return true;
}

bool AbstractDynamicMemPool::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) {
  MS_LOG(DEBUG) << "Wait event for task id on stream : " << task_id_on_stream << ", user stream id : " << user_stream_id
                << ", memory stream id : " << memory_stream_id << ".";
  LockGuard lock(lock_);
  auto key = std::make_pair(user_stream_id, memory_stream_id);
  auto iter = stream_pair_mem_bufs_.find(key);
  if (iter == stream_pair_mem_bufs_.end()) {
    return false;
  }

  auto mem_bufs_ = iter->second;
  for (const auto &mem_buf : mem_bufs_) {
    mem_buf->WaitEvent(task_id_on_stream, user_stream_id);
    // Remove event and try to free memory.
    if (mem_buf->IsEventNotUsed()) {
      (void)iter->second.erase(mem_buf);
      mem_stat_.used_by_event_size_ -= mem_buf->size_;
      if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
        // Force clear all mem bufs.
        for (auto &stream_pair_mem_bufs : stream_pair_mem_bufs_) {
          (void)stream_pair_mem_bufs.second.erase(mem_buf);
        }
        (void)DoFreeTensorMem(mem_buf->addr_);
      }
    }
  }
  return true;
}

bool AbstractDynamicMemPool::WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) {
  MS_LOG(DEBUG) << "Wait event for task id on stream : " << task_id_on_stream
                << ", memory stream id : " << memory_stream_id << ".";
  LockGuard lock(lock_);
  for (auto &stream_pair_mem_bufs : stream_pair_mem_bufs_) {
    const auto &[user_stream, memory_stream] = stream_pair_mem_bufs.first;
    if (memory_stream != memory_stream_id) {
      continue;
    }
    auto mem_bufs = stream_pair_mem_bufs.second;
    for (const auto &mem_buf : mem_bufs) {
      mem_buf->WaitEvent(task_id_on_stream, user_stream);
      // Remove event and try to free memory.
      if (mem_buf->IsEventNotUsed()) {
        (void)stream_pair_mem_bufs.second.erase(mem_buf);
        mem_stat_.used_by_event_size_ -= mem_buf->size_;
        if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
          // Force clear all mem bufs.
          for (auto &kv : stream_pair_mem_bufs_) {
            (void)kv.second.erase(mem_buf);
          }
          (void)DoFreeTensorMem(mem_buf->addr_);
        }
      }
    }
  }
  return true;
}

bool AbstractDynamicMemPool::SyncAllEvents() {
  MS_LOG(DEBUG) << "Sync all events, stream_pair_addresses_ size : " << stream_pair_mem_bufs_.size() << ".";
  LockGuard lock(lock_);
  return DoSyncAllEvents();
}

bool AbstractDynamicMemPool::DoSyncAllEvents() {
  if (stream_pair_mem_bufs_.empty()) {
    return false;
  }

  std::set<MemBuf *> carry_event_mem_bufs;
  for (const auto &stream_pair_mem_buf : stream_pair_mem_bufs_) {
    for (const auto &mem_buf : stream_pair_mem_buf.second) {
      (void)carry_event_mem_bufs.emplace(mem_buf);
    }
  }
  for (auto &mem_buf : carry_event_mem_bufs) {
    if (mem_buf->SyncAllEvents() && mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
      (void)DoFreeTensorMem(mem_buf->addr_);
    }
  }

  stream_pair_mem_bufs_.clear();
  return true;
}

void AbstractDynamicMemPool::DefragMemory() {
  MS_LOG(DEBUG) << "Try to defrag memory.";
  LockGuard lock(lock_);
  if (!enable_vmm_) {
    MS_LOG(DEBUG) << "Skip defrag memory since vmm is not enabled.";
    return;
  }

  if (eager_free_count_ == 0) {
    MS_LOG(DEBUG) << "Exit defrag memory since eager free count is 0.";
    return;
  }
  if (last_eager_free_count_ == eager_free_count_) {
    MS_LOG(DEBUG) << "Exit defrag memory since last eager free count equals to eager free count : "
                  << last_eager_free_count_ << ".";
    return;
  }

  MS_LOG(INFO) << "Defrag memory start.";
  if (!SyncAllStreams()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Sync all streams failed.";
    return;
  }
  const auto [eager_free_size, real_free_size] = FreeIdleMemsByEagerFree();
  MS_LOG(INFO) << "Defrag memory, eager_free_size : " << eager_free_size << ", real_free_size : " << real_free_size
               << ".";
  last_eager_free_count_ = eager_free_count_;
}

namespace {
std::string GetRankID() {
  uint32_t rank_id = 0;
#if !defined(BUILD_LITE)
  if (distributed::collective::CollectiveManager::instance()->initialized()) {
    rank_id = CommManager::GetInstance().GetRank();
  }
#endif
  return std::to_string(rank_id);
}

std::string GetPath() {
  std::string path;
  auto &&ms_context = MsContext::GetInstance();
  path = ms_context->get_param<std::string>(MS_CTX_PROF_MEM_OUTPUT_PATH);
  if (path.empty()) {
    path = "./";
  }
  auto enable_hccl = ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL);
  if (enable_hccl) {
    path += "/rank_" + GetRankID() + "/";

    auto path_opt = Common::CreatePrefixPath(path);
    if (!path_opt.has_value()) {
      MS_LOG(ERROR) << "Create path : " << path << " failed.";
    }
  } else {
    path += "/";
  }
  return path;
}

void SplitAndDumpLog(const std::string &log_str) {
  const char *delim = "\n";
  char *next_token;
#ifdef _MSC_VER
  const char *token = strtok_s(const_cast<char *>(log_str.c_str()), delim, &next_token);
#else
  const char *token = strtok_r(const_cast<char *>(log_str.c_str()), delim, &next_token);
#endif
  size_t count = 0;
  const size_t max_count = 1000;
  while (token && (count++ < max_count)) {
    MS_LOG(INFO) << token;
#ifdef _MSC_VER
    token = strtok_s(NULL, delim, &next_token);
#else
    token = strtok_r(NULL, delim, &next_token);
#endif
  }
}
};  // namespace

void AbstractDynamicMemPool::DumpDynamicMemPoolStateInfo() { SplitAndDumpLog(DynamicMemPoolStateInfo()); }

std::string AbstractDynamicMemPool::DynamicMemPoolStateInfo() const {
  std::stringstream ss;
  // Classify mem buf and stat mem buf state info.
  size_t mem_buf_used_stat[kAllocatorTypeNum] = {0};
  struct AddrComparator {
    bool operator()(MemBuf *const &left, MemBuf *const &right) const { return left->addr_ < right->addr_; }
  };
  std::map<MemBufAllocator *, std::set<MemBuf *, AddrComparator>> allocator_mem_bufs;
  for (const auto &addr_mem_buf_allocator : addr_mem_buf_allocators_) {
    const auto allocator = addr_mem_buf_allocator.second.second;
    const auto mem_buf = addr_mem_buf_allocator.second.first;
    mem_buf_used_stat[static_cast<int>(mem_buf->alloc_type_)] += mem_buf->size_;
    auto &mem_bufs = allocator_mem_bufs[allocator];
    (void)mem_bufs.insert(mem_buf);
  }
  for (const auto &[allocator, mem_bufs] : allocator_mem_bufs) {
    ss << "\tIn used mem buf info for " << allocator->BriefInfo() << ", mem_bufs size : " << mem_bufs.size() << "\n";
  }

  for (const auto &stream_id_allocator : stream_id_allocators_) {
    ss << "stream id : " << stream_id_allocator.first.second << ", is persistent : " << stream_id_allocator.first.first
       << "\n";
    ss << stream_id_allocator.second->DumpStateInfo();
  }

  ss << "The dynamic memory pool stat info : " << mem_stat_.ToReadableString()
     << ", actual peak used mem:" << ActualPeakStatistics() / kMBToByte
     << "M. Weight used size:" << mem_buf_used_stat[static_cast<int>(AllocatorType::kWeight)] / kMBToByte
     << "M, constant value used size:" << mem_buf_used_stat[static_cast<int>(AllocatorType::kConstantValue)] / kMBToByte
     << "M, kernel output used size:" << mem_buf_used_stat[static_cast<int>(AllocatorType::kKernelOutput)] / kMBToByte
     << "M, other used size:" << mem_buf_used_stat[static_cast<int>(AllocatorType::kOther)] / kMBToByte << "M.\n";
  return ss.str();
}

void AbstractDynamicMemPool::DumpDynamicMemPoolDebugInfo() {
  std::stringstream ss;
  ss << "The dynamic memory pool stat info : " << mem_stat_.ToReadableString() << "\n";

  struct AddrComparator {
    bool operator()(MemBuf *const &left, MemBuf *const &right) const { return left->addr_ < right->addr_; }
  };
  std::map<MemBufAllocator *, std::set<MemBuf *, AddrComparator>> allocator_mem_bufs;
  for (const auto &addr_mem_buf_allocator : addr_mem_buf_allocators_) {
    const auto allocator = addr_mem_buf_allocator.second.second;
    const auto mem_buf = addr_mem_buf_allocator.second.first;
    auto &mem_bufs = allocator_mem_bufs[allocator];
    (void)mem_bufs.insert(mem_buf);
  }

  for (const auto &[allocator, mem_bufs] : allocator_mem_bufs) {
    ss << "\tIn used mem buf info for " << allocator->BriefInfo() << ", mem_bufs size : " << mem_bufs.size() << "\n";
    for (const auto mem_buf : mem_bufs) {
      ss << "\t\t" << mem_buf->ToJson() << "\n";
    }
  }

  for (const auto &stream_id_allocator : stream_id_allocators_) {
    ss << "stream id : " << stream_id_allocator.first.second << ", is persistent : " << stream_id_allocator.first.first
       << "\n";
    ss << stream_id_allocator.second->DumpDebugInfo();
  }

  // Write info file.
  static std::atomic<size_t> file_count = 0;
  auto file_path_str = GetPath() + "/MemoryDebugInfo" + std::to_string(file_count++) + ".txt";
  auto file_path_opt = Common::CreatePrefixPath(file_path_str);
  ChangeFileMode(file_path_opt.value(), S_IWUSR | S_IRUSR);
  std::ofstream debug_info_file(file_path_opt.value());
  debug_info_file << ss.str();
  debug_info_file.close();
}

const std::pair<size_t, size_t> AbstractDynamicMemPool::FreeIdleMemsByEagerFree() {
  MS_LOG(INFO) << "Free idle mems by eager free start, allocator size : " << stream_id_allocators_.size() << ".";
  eager_free_count_++;

  size_t total_eager_free_size = 0;
  size_t total_real_free_size = 0;
  for (auto &stream_id_allocator : stream_id_allocators_) {
    const auto [eager_free_size, real_free_size] = stream_id_allocator.second->FreeIdleMemsByEagerFree();
    total_eager_free_size += eager_free_size;
    total_real_free_size += real_free_size;
  }

  size_t not_free_size =
    total_eager_free_size > total_real_free_size ? (total_eager_free_size - total_real_free_size) : 0;
  if (total_real_free_size >= kGBToByte) {
    MS_LOG(WARNING) << "Eager free count : " << eager_free_count_ << ", free memory : " << total_eager_free_size
                    << ", real free : " << total_real_free_size << ", not free : " << not_free_size << ".";
  } else {
    MS_LOG(INFO) << "Eager free count : " << eager_free_count_ << ", free memory : " << total_eager_free_size
                 << ", real free : " << total_real_free_size << ", not free : " << not_free_size << ".";
  }

  mem_stat_.eager_free_size_ += total_eager_free_size;
  return {total_eager_free_size, total_real_free_size};
}

// The statistics information.
size_t AbstractDynamicMemPool::TotalMemStatistics() const { return mem_stat_.alloc_size_; }

size_t AbstractDynamicMemPool::TotalUsedMemStatistics() const { return mem_stat_.used_size_; }

size_t AbstractDynamicMemPool::TotalUsedByEventMemStatistics() const { return mem_stat_.used_by_event_size_; }

size_t AbstractDynamicMemPool::TotalIdleMemStatistics() const { return mem_stat_.IdleSize(); }

size_t AbstractDynamicMemPool::TotalEagerFreeMemStatistics() const { return mem_stat_.eager_free_size_; }

size_t AbstractDynamicMemPool::UsedMemPeakStatistics() const { return mem_stat_.peak_size_; }

size_t AbstractDynamicMemPool::MaxMemAllocatedStatistics() const { return mem_stat_.temp_peak_size_; }

size_t AbstractDynamicMemPool::MaxMemReservedStatistics() const {
  return mem_stat_.alloc_size_ - mem_stat_.temp_alloc_size_;
}

size_t AbstractDynamicMemPool::ActualPeakStatistics() const {
  if (IsEnableVmm()) {
    return GetVmmUsedMemSize();
  }

  size_t peak_size = 0;
  for (auto &stream_id_allocator : stream_id_allocators_) {
    peak_size += stream_id_allocator.second->ActualPeakSize();
  }
  return peak_size;
}

std::unordered_map<std::string, std::size_t> AbstractDynamicMemPool::BlockCountsStatistics() const {
  LockGuard lock(lock_);
  size_t persistent_block_count = 0;
  size_t common_block_count = 0;
  for (const auto &iter : stream_id_allocators_) {
    if (iter.first.first) {
      persistent_block_count += iter.second->mem_blocks_.size();
    } else {
      common_block_count += iter.second->mem_blocks_.size();
    }
  }
  std::unordered_map<std::string, size_t> block_counts;
  block_counts[kPersistentMemPoolType] = persistent_block_count;
  block_counts[kCommonMemPoolType] = common_block_count;
  return block_counts;
}

std::unordered_map<std::string, std::size_t> AbstractDynamicMemPool::BlockUnitSizeStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<std::string, size_t> block_units;
  block_units[kPersistentMemPoolType] = persist_unit_size_;
  block_units[kCommonMemPoolType] = common_unit_size_;
  return block_units;
}

std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
AbstractDynamicMemPool::CommonMemBlocksInfoStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> block_infos;
  for (const auto &iter : stream_id_allocators_) {
    if (!iter.first.first) {
      const auto &mem_blocks = iter.second->mem_blocks_;
      for (const auto mem_block : mem_blocks) {
        std::unordered_map<std::string, size_t> block_info;
        block_info[kBlockMemorySize] = mem_block->size_;
        block_info[kBlockStreamId] = mem_block->stream_id_;
        block_infos[(std::string *)(mem_block->addr_)] = block_info;
      }
    }
  }
  return block_infos;
}

std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>
AbstractDynamicMemPool::PersistentMemBlocksInfoStatistics() const {
  LockGuard lock(lock_);
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> block_infos;
  for (const auto &iter : stream_id_allocators_) {
    if (iter.first.first) {
      const auto &mem_blocks = iter.second->mem_blocks_;
      for (const auto mem_block : mem_blocks) {
        std::unordered_map<std::string, size_t> block_info;
        block_info[kBlockMemorySize] = mem_block->size_;
        block_info[kBlockStreamId] = mem_block->stream_id_;
        block_infos[(std::string *)(mem_block->addr_)] = block_info;
      }
    }
  }
  return block_infos;
}

void AbstractDynamicMemPool::ResetMaxMemReserved() {
  LockGuard lock(lock_);
  mem_stat_.temp_alloc_size_ = mem_stat_.alloc_size_;
}

void AbstractDynamicMemPool::ResetMaxMemAllocated() {
  LockGuard lock(lock_);
  mem_stat_.temp_used_size_ = mem_stat_.used_size_;
  mem_stat_.temp_used_by_event_size_ = mem_stat_.used_by_event_size_;
  mem_stat_.temp_peak_size_ = 0;
}
}  // namespace device
}  // namespace mindspore
