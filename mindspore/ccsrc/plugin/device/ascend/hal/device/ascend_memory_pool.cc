/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/hal/device/ascend_memory_pool.h"

#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "include/backend/distributed/collective/collective_manager.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "plugin/device/ascend/hal/device/ascend_vmm_adapter.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#ifdef ENABLE_DEBUGGER
#include "plugin/device/cpu/hal/profiler/cpu_profiling.h"
#endif
#include "utils/log_adapter.h"

namespace mindspore {
namespace device {
namespace ascend {
DefaultAscendMemoryPool::DefaultAscendMemoryPool() {
  MS_LOG(DEBUG) << "DefaultAscendMemoryPool constructed.";
  SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

AscendMemoryTimeEvent::AscendMemoryTimeEvent(int32_t device_id, const MemoryTimeEventPtr &memory_time_event)
    : BaseReportData(device_id, "memory_time_event"), memory_time_event_(memory_time_event) {
  stream_ptr_ = AscendStreamMng::GetInstance().GetStream(memory_time_event_->stream_id_);
}

namespace {
template <typename T>
void EncodeIntoUInt8(T data, std::vector<uint8_t> *result) {
  for (size_t i = 0; i < sizeof(T); i++) {
    result->push_back((static_cast<size_t>(data) >> (i * 8)) & 0xff);
  }
}

void EncodeStringIntoUInt8(std::string str, std::vector<uint8_t> *result) {
  uint16_t str_type = static_cast<uint16_t>(profiler::ascend::OpRangeDataType::NAME);
  for (size_t i = 0; i < sizeof(uint16_t); ++i) {
    result->push_back((str_type >> (i * 8)) & 0xff);
  }
  uint32_t length = str.size();
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    result->push_back((length >> (i * 8)) & 0xff);
  }
  result->insert(result->end(), str.begin(), str.end());
}

static uint64_t GetTid() {
#ifdef __GNUC__
  static thread_local uint64_t tid = static_cast<uint64_t>(syscall(SYS_gettid));
#else
  static thread_local uint64_t tid = static_cast<uint64_t>(GetCurrentThreadId());
#endif
  return tid;
}

static uint64_t GetPid() {
  static thread_local uint64_t pid = static_cast<uint64_t>(getpid());
  return pid;
}

int32_t GetDeviceId() {
  int32_t device_id = 0;
#if !defined(BUILD_LITE)
  device_id = static_cast<int32_t>(distributed::collective::CollectiveManager::instance()->local_rank_id());
#endif
  return device_id;
}

void FillTidAndPid(const std::unique_ptr<AscendMemoryTimeEvent> &ascend_mmemory_time_event) {
  ascend_mmemory_time_event->tid_ = GetTid();
  ascend_mmemory_time_event->pid_ = GetPid();
  MS_LOG(DEBUG) << "after fill time event info : " << ascend_mmemory_time_event->ToJson();
}
}  // namespace

std::vector<uint8_t> AscendMemoryTimeEvent::encode() {
  std::vector<uint8_t> result;
  EncodeIntoUInt8(device_id, &result);
  EncodeIntoUInt8(memory_time_event_->created_at_, &result);
  EncodeIntoUInt8(reinterpret_cast<size_t>(memory_time_event_->addr_), &result);
  EncodeIntoUInt8(memory_time_event_->size_, &result);
  EncodeIntoUInt8(memory_time_event_->from_persistent_, &result);
  EncodeIntoUInt8(memory_time_event_->is_persistent_, &result);
  EncodeIntoUInt8(memory_time_event_->stream_id_, &result);
  EncodeIntoUInt8(memory_time_event_->run_mode_, &result);
  EncodeIntoUInt8(memory_time_event_->used_size_, &result);
  EncodeIntoUInt8(memory_time_event_->peak_size_, &result);
  EncodeIntoUInt8(memory_time_event_->alloc_size_, &result);
  EncodeIntoUInt8(memory_time_event_->used_by_event_size_, &result);
  EncodeIntoUInt8(memory_time_event_->eager_free_size_, &result);
  EncodeStringIntoUInt8(memory_time_event_->owner_, &result);
  EncodeIntoUInt8(memory_time_event_->alloc_type_, &result);
  EncodeIntoUInt8(reinterpret_cast<size_t>(stream_ptr_), &result);
  EncodeIntoUInt8(tid_, &result);
  EncodeIntoUInt8(pid_, &result);
  return result;
}

DefaultEnhancedAscendMemoryPool::DefaultEnhancedAscendMemoryPool(const DefaultAscendMemoryPoolPtr &instance)
    : instance_(instance) {
  MS_LOG(INFO) << "DefaultEnhancedAscendMemoryPool constructed.";
  instance_->SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

void DefaultEnhancedAscendMemoryPool::ReleaseDeviceRes() {
  MS_LOG(INFO) << "Start release device res.";
  instance_->ReleaseDeviceRes();
  tracker::MemTrackerManager::GetInstance().Dump();
  if (instance_->IsEnableTimeEvent()) {
    profiler::ascend::ProfilingDataDumper::GetInstance().Flush();
  }
}

DeviceMemPtr DefaultEnhancedAscendMemoryPool::AllocTensorMem(size_t size, bool from_persistent_mem, bool need_recycle,
                                                             uint32_t stream_id) {
  size_t align_size = AlignMemorySize(size);
  MS_LOG(DEBUG) << "Allocate tensor mem, size : " << size << ", align_size : " << align_size
                << ", from_persistent_mem : " << from_persistent_mem << ", need_recycle : " << need_recycle
                << ", stream_id : " << stream_id << ".";
  LockGuard lock(instance_->lock());
  const auto [mem_buf, allocator] = instance_->AllocMemBuf(align_size, from_persistent_mem, stream_id);
  if (mem_buf == nullptr) {
    MS_LOG(DEBUG) << "Allocate tensor mem, return nullptr.";
    // Dump mem pool state info and debug info when alloc tensor failed.
    DumpDynamicMemPoolStateInfo();
    DumpDynamicMemPoolDebugInfo();
    return nullptr;
  }

  mem_buf->SetDebugInfo();
  instance_->addr_mem_buf_allocators().emplace(mem_buf->addr_, std::make_pair(mem_buf, allocator));
  auto device_addr = mem_buf->addr_;

  instance_->ReportMemoryPoolInfo();

  // Adapt for dry run.
  if (common::IsNeedProfileMemory()) {
    MS_LOG(WARNING) << "Need Profile Memory, Memory pool alloc, total mem: " << TotalMemStatistics()
                    << ", peak mem: " << UsedMemPeakStatistics() << ", in use mem: " << TotalUsedMemStatistics()
                    << ", used by event mem: " << TotalUsedByEventMemStatistics()
                    << ", device address addr: " << device_addr << ", size: " << align_size
                    << ", from persistent mem: " << from_persistent_mem << ", need recycle: " << need_recycle << ".";
  }

  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    tracker::CALL_MEMORY_TRACKER(AllocMemBlock, device_addr, mem_buf->size_, GetMemoryPoolType(),
                                 ActualPeakStatistics(), TotalUsedMemStatistics(), TotalMemStatistics(), stream_id);
  }

  // Time line process.
  if (instance_->IsEnableTimeEvent()) {
    int32_t device_id = GetDeviceId();
    auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(
      device_id, instance_->GenAllocateMemoryTimeEvent(mem_buf->addr_, mem_buf->size_, stream_id, from_persistent_mem,
                                                       allocator->is_persistent()));
    ascend_memory_time_event->stream_ptr_ =
      AscendStreamMng::GetInstance().GetStream(ascend_memory_time_event->memory_time_event_->stream_id_);
    FillTidAndPid(ascend_memory_time_event);
    profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
  }

  MS_LOG(DEBUG) << "Allocate tensor mem, return : " << mem_buf->ToJson()
                << ", stat : " << instance_->mem_stat().ToJson() << ".";
  return device_addr;
}

std::vector<DeviceMemPtr> DefaultEnhancedAscendMemoryPool::AllocContinuousTensorMem(
  const std::vector<size_t> &size_list, uint32_t stream_id) {
  MS_LOG(DEBUG) << "Alloc continuous tensor mem, stream id : " << stream_id << ".";
  const auto &continuous_addrs = instance_->AllocContinuousTensorMem(size_list, stream_id);
  if (continuous_addrs.size() != size_list.size()) {
    return continuous_addrs;
  }
  if (continuous_addrs.size() == 1 && continuous_addrs[0] == nullptr) {
    return continuous_addrs;
  }

  for (size_t i = 0; i < continuous_addrs.size(); i++) {
    if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
      tracker::CALL_MEMORY_TRACKER(AllocMemBlock, continuous_addrs[i], size_list[i], GetMemoryPoolType(),
                                   ActualPeakStatistics(), TotalUsedMemStatistics(), TotalMemStatistics(), stream_id);
    }

    if (instance_->IsEnableTimeEvent()) {
      int32_t device_id = GetDeviceId();
      auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(
        device_id, instance_->GenAllocateMemoryTimeEvent(continuous_addrs[i], size_list[i], stream_id, false, false));
      ascend_memory_time_event->stream_ptr_ =
        AscendStreamMng::GetInstance().GetStream(ascend_memory_time_event->memory_time_event_->stream_id_);
      FillTidAndPid(ascend_memory_time_event);
      profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
    }
  }
  return continuous_addrs;
}

void DefaultEnhancedAscendMemoryPool::FreeTensorMem(const DeviceMemPtr &device_addr) {
  MS_LOG(DEBUG) << "Free tensor mem, device addr : " << device_addr << ".";
  instance_->FreeTensorMem(device_addr);
}

bool DefaultEnhancedAscendMemoryPool::DoFreeTensorMem(const DeviceMemPtr &device_addr) {
  void *enhanced_device_addr = device_addr;
  bool ret = instance_->DoFreeTensorMem(device_addr);
  if (ret) {
    instance_->ReportMemoryPoolInfo();

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

    if (instance_->IsEnableTimeEvent()) {
      int32_t device_id = GetDeviceId();
      auto time_event = GenFreeMemoryTimeEvent(enhanced_device_addr);
      auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(device_id, time_event);
      FillTidAndPid(ascend_memory_time_event);
      profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
    }
  }
  MS_LOG(DEBUG) << "Do free tensor mem : " << enhanced_device_addr << ", ret : " << ret << ".";
  return ret;
}

void DefaultEnhancedAscendMemoryPool::FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                         const std::vector<DeviceMemPtr> &keep_addrs,
                                                         const std::vector<size_t> &keep_addr_sizes) {
  MS_LOG(DEBUG) << "Free part tensor mems.";
  LockGuard lock(instance_->lock());
  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    for (const auto &free_addr : free_addrs) {
      tracker::CALL_MEMORY_TRACKER(FreeMemBlock, free_addr, TotalUsedMemStatistics(), TotalMemStatistics());
    }
  }

  if (instance_->IsEnableTimeEvent()) {
    int32_t device_id = GetDeviceId();
    for (const auto &free_addr : free_addrs) {
      auto time_event = GenFreeMemoryTimeEvent(free_addr);
      auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(device_id, time_event);
      FillTidAndPid(ascend_memory_time_event);
      profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
    }
  }

  const auto keep_mem_bufs = instance_->DoFreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
  if (tracker::MemTrackerManager::GetInstance().IsEnabled()) {
    for (const auto &mem_buf : keep_mem_bufs) {
      tracker::CALL_MEMORY_TRACKER(AllocMemBlock, mem_buf->addr_, mem_buf->size_, GetMemoryPoolType(),
                                   ActualPeakStatistics(), TotalUsedMemStatistics(), TotalMemStatistics(),
                                   mem_buf->stream_id_);
    }
  }
  if (instance_->IsEnableTimeEvent()) {
    int32_t device_id = GetDeviceId();
    for (const auto &mem_buf : keep_mem_bufs) {
      auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(
        device_id,
        instance_->GenAllocateMemoryTimeEvent(mem_buf->addr_, mem_buf->size_, mem_buf->stream_id_, false, false));
      ascend_memory_time_event->stream_ptr_ =
        AscendStreamMng::GetInstance().GetStream(ascend_memory_time_event->memory_time_event_->stream_id_);
      FillTidAndPid(ascend_memory_time_event);
      profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
    }
  }
}

void DefaultEnhancedAscendMemoryPool::DefragMemory() {
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

  instance_->DefragMemory();
}

void DefaultEnhancedAscendMemoryPool::DumpDynamicMemPoolStateInfo() {
  const auto &state_info = instance_->DynamicMemPoolStateInfo();
  static bool is_enable_memory_statistics = common::IsEnableRuntimeConfig(common::kRuntimeMemoryStat);
  if (is_enable_memory_statistics) {
    std::cout << "[MS_RUNTIME_PROF]" << state_info << std::endl;
  }
  instance_->DumpDynamicMemPoolStateInfo();
}

const std::pair<size_t, size_t> DefaultEnhancedAscendMemoryPool::FreeIdleMemsByEagerFree() {
  const auto [eager_free_size, real_free_size] = instance_->FreeIdleMemsByEagerFree();
  static bool is_enable_memory_statistics = common::IsEnableRuntimeConfig(common::kRuntimeMemoryStat);
  if (is_enable_memory_statistics) {
    std::cout << "Total eager free memory : " << eager_free_size << ", real free : " << real_free_size << "."
              << std::endl;
  }
  return {eager_free_size, real_free_size};
}

BestFitAscendMemoryPool::BestFitAscendMemoryPool() {
  MS_LOG(WARNING) << "BestFitAscendMemoryPool constructed, older memory allocator is enabled.";
  SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

// Initialize static member in AscendMemoryPool.
AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::pool_ = nullptr;

AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::instance_ = nullptr;

AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::enhanced_instance_ = nullptr;
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
