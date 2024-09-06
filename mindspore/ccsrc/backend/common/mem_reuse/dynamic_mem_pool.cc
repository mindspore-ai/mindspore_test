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

#include "include/backend/mem_reuse/dynamic_mem_pool.h"

#include <numeric>
#include <ostream>

#include "include/backend/mem_reuse/mem_tracker.h"
#include "include/common/utils/utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
thread_local AllocatorDebugInfo DynamicMemAllocatorDebugInfo::debug_info_;

static const std::map<DynamicMemBufStatus, std::string> kBufStatusString = {
  {DynamicMemBufStatus::kMemBufIdle, "idle"},
  {DynamicMemBufStatus::kMemBufUsed, "used"},
  {DynamicMemBufStatus::kMemBufEagerFree, "eager_free"},
  {DynamicMemBufStatus::kMemBufUsedByEvent, "used_by_event"}};

const std::string &DynamicMemBufStatusToString(DynamicMemBufStatus status) { return kBufStatusString.at(status); }

static const std::map<AllocatorType, std::string> kAllocatorTypeString = {
  {AllocatorType::kWeight, "weight"},
  {AllocatorType::kConstantValue, "constant value"},
  {AllocatorType::kKernelOutput, "kernel output"},
  {AllocatorType::kGraphOutput, "graph output"},
  {AllocatorType::kWorkspace, "workspace"},
  {AllocatorType::kOther, "other"},
};

const std::string &AllocatorTypeToString(AllocatorType allocator_type) {
  return kAllocatorTypeString.at(allocator_type);
}

void DynamicMemBlock::update_border_addr(DeviceMemPtr left_addr, DeviceMemPtr right_addr) {
  if (min_addr_ == nullptr) {
    min_addr_ = left_addr;
  } else {
    min_addr_ = std::min(min_addr_, left_addr);
  }
  if (max_addr_ == nullptr) {
    max_addr_ = right_addr;
  } else {
    max_addr_ = std::max(max_addr_, right_addr);
  }
}

size_t DynamicMemBlock::get_actual_peak() {
  if (min_addr_ == nullptr || max_addr_ == nullptr) {
    return 0;
  }
  int64_t actual_memory = reinterpret_cast<uint8_t *>(max_addr_) - reinterpret_cast<uint8_t *>(min_addr_);
  return actual_memory;
}

size_t MemStatusManager::CalActualPeak() {
  if (mem_block_insertion_order_.empty()) {
    return 0;
  }
  size_t actual_peak = total_block_size_;
  const auto &end_block = mem_block_insertion_order_.back();
  MS_EXCEPTION_IF_NULL(end_block);
  actual_peak -= end_block->size();
  actual_peak += end_block->get_actual_peak();
  return actual_peak;
}

bool EventBase::RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id, const DeviceEventPtr &event) {
  MS_EXCEPTION_IF_NULL(event);
  if (events_ == nullptr) {
    events_ = std::make_shared<std::unordered_map<uint32_t, std::shared_ptr<std::list<TaskIdOnStreamEvent>>>>();
  }
  std::shared_ptr<std::list<TaskIdOnStreamEvent>> event_list = nullptr;
  auto iter = events_->find(user_stream_id);
  if (iter == events_->end()) {
    event_list = std::make_shared<std::list<TaskIdOnStreamEvent>>();
    (void)events_->emplace(user_stream_id, event_list);
  } else {
    event_list = iter->second;
    MS_EXCEPTION_IF_NULL(event_list);
  }
  (void)event_list->emplace_back(task_id_on_stream, event);
  return true;
}

bool EventBase::WaitEvent(uint32_t task_id_on_stream, uint32_t user_stream_id) {
  if (events_ == nullptr) {
    return false;
  }
  auto iter = events_->find(user_stream_id);
  if (iter == events_->end()) {
    return false;
  }
  auto &event_list = iter->second;
  MS_EXCEPTION_IF_NULL(event_list);
  // Pop all element in list that not bigger than task_id_on_stream.
  while (!event_list->empty() && event_list->front().first <= task_id_on_stream) {
    event_list->pop_front();
  }
  // Remove list if event list is empty.
  if (event_list->empty()) {
    events_->erase(iter);
  }
  return true;
}

bool EventBase::IsEventNotUsed() { return events_ == nullptr ? true : events_->empty(); }

bool EventBase::SyncAllEvents() {
  if (IsEventNotUsed()) {
    return false;
  }

  for (auto iter = events_->begin(); iter != events_->end();) {
    auto &event_list = iter->second;
    MS_EXCEPTION_IF_NULL(event_list);
    for (auto list_iter = event_list->begin(); list_iter != event_list->end();) {
      auto &event = list_iter->second;
      // Sync event if event is not arrived.
      if (!event->QueryEvent()) {
        event->SyncEvent();
      }
      list_iter = event_list->erase(list_iter);
    }
    if (event_list->empty()) {
      // list is empty, erase list in map.
      iter = events_->erase(iter);
    } else {
      MS_LOG(INTERNAL_EXCEPTION) << "Event list is not empty.";
    }
  }
  return events_->empty();
}

void MemStatusManager::AddMemBlock(const DynamicMemBlockPtr &mem_block, uint32_t stream_id) {
  auto iter = mem_blocks_.find(stream_id);
  if (iter != mem_blocks_.end()) {
    DoAddMemBlock(mem_block, &iter->second);
  } else {
    (void)mem_blocks_.emplace(stream_id, std::vector<DynamicMemBlockPtr>{mem_block});
  }

  DoAddMemBlock(mem_block, &mem_block_list_);
  mem_block_insertion_order_.emplace_back(mem_block);
  total_block_size_ += mem_block->size();
}

void MemStatusManager::DoAddMemBlock(const DynamicMemBlockPtr &mem_block,
                                     std::vector<DynamicMemBlockPtr> *mem_block_list) {
  auto iter = std::upper_bound(mem_block_list->begin(), mem_block_list->end(), mem_block->device_addr(),
                               [](const DeviceMemPtr &device_addr, const DynamicMemBlockPtr &mem_block) {
                                 return device_addr < mem_block->device_addr();
                               });
  (void)mem_block_list->insert(iter, mem_block);
}

SizeMapMemBuf &MemStatusManager::GetOrCreateMemBufMap(uint32_t stream_id, DynamicMemBufStatus status) {
  return mem_bufs_[std::make_pair(stream_id, status)];
}

void MemStatusManager::AddMemBuf(const DynamicMemBufPtr &mem_buf) {
  auto key = std::make_pair(mem_buf->stream_id_, mem_buf->status_);
  auto &mem_buf_map = mem_bufs_[key];
  (void)mem_buf_map.emplace(mem_buf->size_, mem_buf);
}

void MemStatusManager::RemoveMemBuf(const DynamicMemBufPtr &mem_buf) {
  auto key = std::make_pair(mem_buf->stream_id_, mem_buf->status_);
  auto &mem_buf_map = mem_bufs_[key];
  auto &&iter = mem_buf_map.equal_range(mem_buf->size_);
  while (iter.first != iter.second) {
    if (iter.first->second->device_addr_ == mem_buf->device_addr_) {
      (void)mem_buf_map.erase(iter.first);
      return;
    }
    (void)iter.first++;
  }
  MS_LOG(INTERNAL_EXCEPTION) << "Remove mem buf failed, address : " << mem_buf->device_addr_ << ".";
}

void MemStatusManager::Clear() noexcept {
  mem_blocks_.clear();
  mem_block_list_.clear();
  mem_bufs_.clear();
}

const DeviceState MemStatusManager::DumpMemBlockDebugInfo(const std::string &mem_type) {
  DeviceState device_state;
  // Dump the memory block info and memory buf info.
  MS_LOG(WARNING) << mem_type << " all mem_block info: counts[" << mem_block_list_.size() << "].";
  for (auto iter = mem_block_list_.begin(); iter != mem_block_list_.end(); ++iter) {
    device_state.total_mem_size_ += (*iter)->size();
    auto mem_buf_map = (*iter)->block_all_mem_buf_map_;
    MS_LOG(WARNING) << " MemBlock info: number[" << iter - mem_block_list_.begin() << "] mem_buf_counts["
                    << mem_buf_map.size() << "] base_address[" << (*iter)->device_addr() << "] block_size["
                    << (*iter)->size() << "] stream id[" << (*iter)->stream_id_ << "].";
    for (auto iter_mem_buf = mem_buf_map.begin(); iter_mem_buf != mem_buf_map.end(); ++iter_mem_buf) {
      auto mem_buf = iter_mem_buf->second;
      MS_EXCEPTION_IF_NULL(mem_buf);
      if (mem_buf->status_ == DynamicMemBufStatus::kMemBufIdle) {
        device_state.total_idle_mem_size_ += mem_buf->size_;
      } else if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsed) {
        device_state.total_used_mem_size_ += mem_buf->size_;
      } else if (mem_buf->status_ == DynamicMemBufStatus::kMemBufEagerFree) {
        device_state.total_eager_free_mem_size_ += mem_buf->size_;
      } else if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
        device_state.total_used_by_event_mem_size_ += mem_buf->size_;
      } else {
        MS_LOG(INTERNAL_EXCEPTION) << "Unknown mem buf status : " << mem_buf->status_ << ".";
      }
      MS_LOG(INFO) << "  MemBuf info: address[" << mem_buf->device_addr_ << "] size[" << mem_buf->size_ << "] status["
                   << DynamicMemBufStatusToString(mem_buf->status_) << "] name["
                   << (mem_buf->allocator_name_.empty() ? "Unknown" : mem_buf->allocator_name_) << "] type["
                   << AllocatorTypeToString(mem_buf->allocator_type_) << "] stream id[" << mem_buf->stream_id_ << "].";
    }
  }
  return device_state;
}
}  // namespace device
}  // namespace mindspore
