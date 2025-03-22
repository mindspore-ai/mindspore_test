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

#include "plugin/res_manager/ascend/mem_manager/ascend_memory_pool.h"

#include <fcntl.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <atomic>
#include <set>

#include "debug/profiler/profiling_data_dumper.h"
#include "debug/profiler/profiling.h"
#include "include/backend/mem_reuse/mem_tracker.h"
#include "include/common/utils/comm_manager.h"
#include "plugin/res_manager/ascend/mem_manager/ascend_vmm_adapter.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "runtime/pipeline/pipeline.h"
#include "include/common/runtime_conf/runtime_conf.h"
#include "utils/distributed_meta.h"

namespace mindspore {
namespace device {
namespace ascend {
constexpr size_t kByteOffset = 8;

struct AscendMemoryTimeEvent : profiler::ascend::BaseReportData {
  explicit AscendMemoryTimeEvent(int32_t device_id, const MemoryTimeEventPtr &memory_time_event);
  virtual ~AscendMemoryTimeEvent() = default;

  std::vector<uint8_t> encode() override;

  uint64_t tid_{0};

  uint64_t pid_{0};

  void *stream_ptr_{nullptr};

  MemoryTimeEventPtr memory_time_event_{nullptr};

  std::string ToJson() {
    JsonBuilder builder;
    builder.Append("tid_", tid_);
    builder.Append("pid_", pid_);
    builder.Append("stream_ptr_", stream_ptr_);
    builder.Append("memory_time_event_", memory_time_event_ ? memory_time_event_->ToJson() : nullptr);
    return builder.ToString();
  }
};
using AscendMemoryTimeEventPtr = std::shared_ptr<AscendMemoryTimeEvent>;

DefaultAscendMemoryPool::DefaultAscendMemoryPool() {
  MS_LOG(DEBUG) << "DefaultAscendMemoryPool constructed.";
  SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

size_t DefaultAscendMemoryPool::EmptyCache() {
  if (IsEnableVmm() || IsEnableEagerFree()) {
    LockGuard lock(AbstractDynamicMemPool::lock());
    AbstractEnhancedDynamicMemPool::WaitPipelineHelper();
    AbstractAscendMemoryPoolSupport::SyncAllStreams();
    AbstractEnhancedDynamicMemPool::FreeIdleMemsByEagerFree();
    return AbstractAscendMemoryPoolSupport::EmptyCache();
  }
  return 0L;
}

AscendMemoryTimeEvent::AscendMemoryTimeEvent(int32_t device_id, const MemoryTimeEventPtr &memory_time_event)
    : BaseReportData(device_id, static_cast<uint32_t>(profiler::ascend::ReportFileType::MEMORY_USAGE)),
      memory_time_event_(memory_time_event) {
  stream_ptr_ = AscendStreamMng::GetInstance().GetStream(memory_time_event_->stream_id_);
}

namespace {
template <typename T>
void EncodeIntoUInt8(T data, std::vector<uint8_t> *result) {
  for (size_t i = 0; i < sizeof(T); i++) {
    result->push_back((static_cast<size_t>(data) >> (i * kByteOffset)) & 0xff);
  }
}

void EncodeStringIntoUInt8(std::string str, std::vector<uint8_t> *result) {
  uint16_t str_type = static_cast<uint16_t>(profiler::ascend::OpRangeDataType::NAME);
  for (size_t i = 0; i < sizeof(uint16_t); ++i) {
    result->push_back((str_type >> (i * kByteOffset)) & 0xff);
  }
  uint32_t length = str.size();
  for (size_t i = 0; i < sizeof(uint32_t); ++i) {
    result->push_back((length >> (i * kByteOffset)) & 0xff);
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
  device_id = static_cast<int32_t>(DistributedMeta::GetInstance()->local_rank_id());
#endif
  return device_id;
}

void FillTidAndPid(const std::unique_ptr<AscendMemoryTimeEvent> &ascend_mmemory_time_event) {
  ascend_mmemory_time_event->tid_ = GetTid();
  ascend_mmemory_time_event->pid_ = GetPid();
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Fill time event info : " << ascend_mmemory_time_event->ToJson() << ".";
}
}  // namespace

std::vector<uint8_t> AscendMemoryTimeEvent::encode() {
  std::vector<uint8_t> result;
  EncodeIntoUInt8<int32_t>(device_id, &result);
  EncodeIntoUInt8<uint64_t>(tid_, &result);
  EncodeIntoUInt8<uint64_t>(pid_, &result);
  EncodeIntoUInt8<uint64_t>(memory_time_event_->created_at_, &result);
  EncodeIntoUInt8<size_t>(reinterpret_cast<size_t>(memory_time_event_->addr_), &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->size_, &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->used_size_, &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->peak_size_, &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->alloc_size_, &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->used_by_event_size_, &result);
  EncodeIntoUInt8<size_t>(memory_time_event_->eager_free_size_, &result);
  EncodeIntoUInt8<size_t>(reinterpret_cast<size_t>(stream_ptr_), &result);
  EncodeIntoUInt8<uint32_t>(memory_time_event_->stream_id_, &result);
  EncodeIntoUInt8<uint8_t>(memory_time_event_->from_persistent_, &result);
  EncodeIntoUInt8<uint8_t>(memory_time_event_->is_persistent_, &result);
  EncodeIntoUInt8<uint8_t>(memory_time_event_->run_mode_, &result);
  EncodeIntoUInt8<uint8_t>(memory_time_event_->alloc_type_, &result);
  EncodeStringIntoUInt8(memory_time_event_->owner_, &result);

  std::vector<uint8_t> tlv_result;
  uint16_t data_type = static_cast<uint16_t>(profiler::ascend::OpRangeDataType::NAME);
  for (size_t i = 0; i < sizeof(uint16_t); i++) {
    (void)tlv_result.emplace_back(data_type >> (i * kByteOffset) & 0xff);
  }
  uint32_t length = result.size();
  for (size_t i = 0; i < sizeof(uint32_t); i++) {
    (void)tlv_result.emplace_back(length >> (i * kByteOffset) & 0xff);
  }
  tlv_result.insert(tlv_result.end(), result.cbegin(), result.cend());
  return tlv_result;
}

DefaultEnhancedAscendMemoryPool::DefaultEnhancedAscendMemoryPool(const DefaultAscendMemoryPoolPtr &instance)
    : instance_(instance) {
  MS_LOG(INFO) << "DefaultEnhancedAscendMemoryPool constructed.";
  instance_->SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

void DefaultEnhancedAscendMemoryPool::ReleaseDeviceRes() {
  MS_LOG(INFO) << "Start release device res.";
  instance_->ReleaseDeviceRes();
  tracker::MemTrackerManager::GetInstance().Dump(rank_id_getter_());
}

DeviceMemPtr DefaultEnhancedAscendMemoryPool::AllocTensorMem(size_t size, bool from_persistent_mem, bool need_recycle,
                                                             uint32_t stream_id) {
  size_t align_size = AlignMemorySize(size);
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Allocate tensor mem, size : " << size << ", align_size : " << align_size
                                       << ", from_persistent_mem : " << from_persistent_mem
                                       << ", need_recycle : " << need_recycle << ", stream_id : " << stream_id << ".";
  LockGuard lock(instance_->lock());
  const auto [mem_buf, allocator] = instance_->AllocMemBuf(align_size, from_persistent_mem, stream_id);
  if (mem_buf == nullptr) {
    MS_LOG(INFO) << "Allocate tensor mem, return nullptr.";
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
  if (IsNeedProfilieMemoryLog()) {
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

  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Allocate tensor mem, return : " << mem_buf->ToJson()
                                       << ", stat info : " << instance_->mem_stat().ToJson() << ".";
  return device_addr;
}

std::vector<DeviceMemPtr> DefaultEnhancedAscendMemoryPool::AllocContinuousTensorMem(
  const std::vector<size_t> &size_list, uint32_t stream_id) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Alloc continuous tensor mem, stream id : " << stream_id << ".";
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
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Free tensor mem, device addr : " << device_addr << ".";
  LockGuard lock(instance_->lock());
  DoFreeTensorMem(device_addr);
}

bool DefaultEnhancedAscendMemoryPool::DoFreeTensorMem(const DeviceMemPtr &device_addr) {
  void *enhanced_device_addr = device_addr;
  bool ret = instance_->DoFreeTensorMem(device_addr);
  if (ret) {
    instance_->ReportMemoryPoolInfo();

    // Adapt for dry run.
    if (IsNeedProfilieMemoryLog()) {
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
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Do free tensor mem : " << enhanced_device_addr << ", return : " << ret
                                       << ".";
  return ret;
}

void DefaultEnhancedAscendMemoryPool::FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                                         const std::vector<DeviceMemPtr> &keep_addrs,
                                                         const std::vector<size_t> &keep_addr_sizes) {
  MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Free part tensor mems.";
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

bool DefaultEnhancedAscendMemoryPool::WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                                                uint32_t memory_stream_id) {
  LockGuard lock(instance_->lock());
  auto key = std::make_pair(user_stream_id, memory_stream_id);
  auto iter = instance_->stream_pair_mem_bufs().find(key);
  if (iter == instance_->stream_pair_mem_bufs().end()) {
    return false;
  }

  auto mem_bufs_ = iter->second;
  for (const auto &mem_buf : mem_bufs_) {
    MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Wait event for : " << mem_buf->ToJson() << ".";
    mem_buf->WaitEvent(task_id_on_stream, user_stream_id);
    // Remove event and try to free memory.
    if (mem_buf->IsEventNotUsed()) {
      instance_->mem_stat().used_by_event_size_ -= mem_buf->size_;
      // Force clear all mem bufs.
      for (auto &stream_pair_mem_bufs : instance_->stream_pair_mem_bufs()) {
        (void)stream_pair_mem_bufs.second.erase(mem_buf);
      }
      if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
        (void)DoFreeTensorMem(mem_buf->addr_);
      }
    }
  }
  return true;
}

bool DefaultEnhancedAscendMemoryPool::WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) {
  LockGuard lock(instance_->lock());
  for (auto &stream_pair_mem_bufs : instance_->stream_pair_mem_bufs()) {
    const auto &[user_stream, memory_stream] = stream_pair_mem_bufs.first;
    if (memory_stream != memory_stream_id) {
      continue;
    }
    auto mem_bufs = stream_pair_mem_bufs.second;
    for (const auto &mem_buf : mem_bufs) {
      MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Wait event for : " << mem_buf->ToJson() << ".";
      mem_buf->WaitEvent(task_id_on_stream, user_stream);
      // Remove event and try to free memory.
      if (mem_buf->IsEventNotUsed()) {
        instance_->mem_stat().used_by_event_size_ -= mem_buf->size_;
        // Force clear all mem bufs.
        for (auto &kv : instance_->stream_pair_mem_bufs()) {
          (void)kv.second.erase(mem_buf);
        }
        if (mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
          (void)DoFreeTensorMem(mem_buf->addr_);
        }
      }
    }
  }
  return true;
}

bool DefaultEnhancedAscendMemoryPool::SyncAllEvents() {
  LockGuard lock(instance_->lock());
  if (stream_pair_mem_bufs().empty()) {
    return false;
  }

  std::set<MemBuf *> carry_event_mem_bufs;
  for (const auto &stream_pair_mem_buf : instance_->stream_pair_mem_bufs()) {
    for (const auto &mem_buf : stream_pair_mem_buf.second) {
      (void)carry_event_mem_bufs.emplace(mem_buf);
    }
  }
  for (auto &mem_buf : carry_event_mem_bufs) {
    if (mem_buf->SyncAllEvents() && mem_buf->status_ == DynamicMemBufStatus::kMemBufUsedByEvent) {
      (void)DoFreeTensorMem(mem_buf->addr_);
    }
  }

  instance_->stream_pair_mem_bufs().clear();
  return true;
}

void DefaultEnhancedAscendMemoryPool::SetRankIdGetter(const std::function<size_t()> &rank_id_getter) {
  instance_->SetRankIdGetter(rank_id_getter);
  if (rank_id_getter != nullptr) {
    rank_id_getter_ = rank_id_getter;
  }
}

BestFitAscendMemoryPool::BestFitAscendMemoryPool() {
  MS_LOG(INFO) << "BestFitAscendMemoryPool constructed, older memory allocator is enabled.";
  SetEnableVmm(AscendVmmAdapter::GetInstance().IsEnabled());
}

void BestFitAscendMemoryPool::ReportMemoryTimeEvent(const MemoryTimeEventPtr &time_event) {
  int32_t device_id = GetDeviceId();
  auto ascend_memory_time_event = std::make_unique<AscendMemoryTimeEvent>(device_id, time_event);
  if (time_event->stream_id_ != UINT32_MAX) {
    ascend_memory_time_event->stream_ptr_ = AscendStreamMng::GetInstance().GetStream(time_event->stream_id_);
  }
  FillTidAndPid(ascend_memory_time_event);
  profiler::ascend::ProfilingDataDumper::GetInstance().Report(std::move(ascend_memory_time_event));
}

// Initialize static member in AscendMemoryPool.
AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::pool_ = nullptr;

AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::instance_ = nullptr;

AbstractAscendMemoryPoolSupportPtr AscendMemoryPool::enhanced_instance_ = nullptr;

AbstractAscendMemoryPoolSupport &AscendMemoryPool::GetInstance() {
  static std::once_flag flag;
  std::call_once(flag, [&]() {
    if (UseOldMemoryPool()) {
      instance_ = std::make_shared<BestFitAscendMemoryPool>();
      enhanced_instance_ = instance_;
    } else {
      const auto &memory_pool = std::make_shared<DefaultAscendMemoryPool>();
      instance_ = memory_pool;
      enhanced_instance_ = std::make_shared<DefaultEnhancedAscendMemoryPool>(memory_pool);
    }
    // Initialize instance and set ptr.
    float init_size = runtime::RuntimeConf::GetInstance()->mem_init_size();
    size_t init_size_byte = FloatToSize(init_size * kGBToByte);
    float increase_size = runtime::RuntimeConf::GetInstance()->mem_block_increase_size();
    size_t increase_size_byte = FloatToSize(increase_size * kGBToByte);
    float max_size = runtime::RuntimeConf::GetInstance()->mem_max_size();
    size_t max_size_byte = FloatToSize(max_size * kGBToByte);
    instance_->Initialize(init_size_byte, increase_size_byte, max_size_byte);
#ifdef ENABLE_DEBUGGER
    // Set memory profiler callback func.
    instance_->SetMemoryProfilerCallback([&]() {
      static auto profiler_inst = profiler::Profiler::GetInstance(kCPUDevice);
      MS_EXCEPTION_IF_NULL(profiler_inst);
      MS_VLOG(VL_RUNTIME_FRAMEWORK_MEMORY) << "Start report memory pool info.";
      if (profiler_inst->GetEnableFlag() && profiler_inst->GetProfileMemoryFlag()) {
        profiler_inst->RecordMemoryPoolInfo(instance_->TotalUsedMemStatistics(), instance_->TotalMemStatistics(),
                                            instance_->TotalUsedByEventMemStatistics());
      }
    });
#endif

    enhanced_instance_->SetRankIdGetter([]() {
      size_t rank_id = SIZE_MAX;
      if (DistributedMeta::GetInstance()->initialized()) {
        rank_id = DistributedMeta::GetInstance()->global_rank_id();
      }
      return rank_id;
    });
    instance_->SetPipelineCallback([]() { runtime::Pipeline::Get().launch_stage()->Wait(); });
    if (!UseEnhancedMemoryPool()) {
      pool_ = instance_;
      device::tracker::CALL_MEMORY_TRACKER(SetEnableMemoryDebugInfo, false);
    } else {
      pool_ = enhanced_instance_;
      device::tracker::CALL_MEMORY_TRACKER(SetEnableMemoryDebugInfo, true);
    }
  });
  return *pool_;
}

void AscendMemoryPool::SetEnhancedMemoryPool(bool enable) {
  MS_LOG(INFO) << "Set enhanced memory pool : " << enable << ".";
  if (enable) {
    pool_ = enhanced_instance_;
  } else {
    pool_ = instance_;
  }
  device::tracker::CALL_MEMORY_TRACKER(SetEnableMemoryDebugInfo, enable);
}

bool AscendMemoryPool::UseOldMemoryPool() {
  if (common::IsDisableAllocConfig(common::kAllocMemoryPool)) {
    return false;
  }
  return IsDisableGeKernel() || common::IsEnableAllocConfig(common::kAllocMemoryPool);
}

// Use enhanced memory pool when enable debug, enable log, enable prof, dry run and so on.
bool AscendMemoryPool::UseEnhancedMemoryPool() {
  bool enable_debugger = false;
#ifdef ENABLE_DEBUGGER
  auto profiler = profiler::Profiler::GetInstance(kCPUDevice);
  if (profiler != nullptr && profiler->GetEnableFlag() && profiler->GetProfileMemoryFlag()) {
    enable_debugger = true;
  }
#endif
  bool enable_debug_log = common::GetEnv("GLOG_v") == "0";
  bool enable_memory_vlog = IS_VLOG_ON(VL_RUNTIME_FRAMEWORK_MEMORY);
  return enable_debugger || enable_debug_log || enable_memory_vlog ||
         common::IsEnableAllocConfig(common::kAllocMemoryTracker) ||
         common::IsEnableRuntimeConfig(common::kRuntimeMemoryStat) || common::IsDryRun();
}

std::string AscendMemoryPool::ParseDebugConfig(std::string input, std::string config) {
  auto pos = input.find(config);
  if (pos == std::string::npos) {
    return "";
  }
  auto config_pos = input.find(",", pos);
  size_t skip_count = config.size() + 1;
  auto config_str = input.substr(pos + skip_count, config_pos - pos - skip_count);
  if (config_str.find("}") != std::string::npos) {
    config_str = config_str.substr(0, config_str.size() - 1);
  }
  // need trim laster
  return config_str;
}
}  // namespace ascend
}  // namespace device
}  // namespace mindspore
