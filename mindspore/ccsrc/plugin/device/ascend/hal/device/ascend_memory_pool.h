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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_POOL_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_POOL_H_

#include <atomic>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/debug/profiler/profiling_data_dumper.h"
#include "include/backend/debug/profiler/profiling.h"
#include "include/backend/mem_reuse/abstract_dynamic_mem_pool.h"
#include "include/backend/mem_reuse/mem_dynamic_allocator.h"
#include "include/backend/visible.h"
#include "plugin/device/ascend/hal/device/abstract_ascend_memory_pool_support.h"
#include "plugin/device/ascend/hal/profiler/ascend_profiling.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"
#include "runtime/runtime_conf/runtime_conf.h"

namespace mindspore {
namespace device {
namespace ascend {

class BACKEND_EXPORT DefaultAscendMemoryPool : public AbstractAscendMemoryPoolSupport,
                                               public AbstractEnhancedDynamicMemPool {
 public:
  DefaultAscendMemoryPool();
  DefaultAscendMemoryPool(const DefaultAscendMemoryPool &) = delete;
  DefaultAscendMemoryPool &operator=(const DefaultAscendMemoryPool &) = delete;
  ~DefaultAscendMemoryPool() override = default;

  std::string GetMemoryPoolType() const override { return "DefaultAscendMemoryPool"; }

  void SetMemPoolBlockSize(size_t available_device_mem_size) override {
    return AbstractAscendMemoryPoolSupport::SetMemPoolBlockSize(available_device_mem_size);
  }

  size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) override {
    return AbstractAscendMemoryPoolSupport::CalMemBlockAllocSize(size, from_persistent_mem, need_recycle);
  }

  const bool IsEnableEagerFree() const override { return AbstractAscendMemoryPoolSupport::IsEnableEagerFree(); }
};
using DefaultAscendMemoryPoolPtr = std::shared_ptr<DefaultAscendMemoryPool>;

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

class BACKEND_EXPORT DefaultEnhancedAscendMemoryPool : public DefaultAscendMemoryPool {
 public:
  explicit DefaultEnhancedAscendMemoryPool(const DefaultAscendMemoryPoolPtr &instance);
  DefaultEnhancedAscendMemoryPool(const DefaultEnhancedAscendMemoryPool &) = delete;
  DefaultEnhancedAscendMemoryPool &operator=(const DefaultEnhancedAscendMemoryPool &) = delete;
  ~DefaultEnhancedAscendMemoryPool() override = default;

  // Wrap enhanced function.
  void Initialize(size_t init_size, size_t increase_size, size_t max_size) override {
    instance_->Initialize(init_size, increase_size, max_size);
  }

  void ReleaseDeviceRes() override;

  DeviceMemPtr AllocTensorMem(size_t size, bool from_persistent_mem = false, bool need_recycle = false,
                              uint32_t stream_id = kDefaultStreamIndex) override;

  std::vector<DeviceMemPtr> AllocContinuousTensorMem(const std::vector<size_t> &size_list,
                                                     uint32_t stream_id = kDefaultStreamIndex) override;

  void FreeTensorMem(const DeviceMemPtr &device_addr) override;

  bool DoFreeTensorMem(const DeviceMemPtr &device_addr) override;

  void FreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs, const std::vector<DeviceMemPtr> &keep_addrs,
                          const std::vector<size_t> &keep_addr_sizes) override;

  std::vector<MemBuf *> DoFreePartTensorMems(const std::vector<DeviceMemPtr> &free_addrs,
                                             const std::vector<DeviceMemPtr> &keep_addrs,
                                             const std::vector<size_t> &keep_addr_sizes) override {
    return instance_->DoFreePartTensorMems(free_addrs, keep_addrs, keep_addr_sizes);
  }

  void DefragMemory() override;

  void DumpDynamicMemPoolStateInfo() override;

  const std::pair<size_t, size_t> FreeIdleMemsByEagerFree() override;

  // Proxy wrapper for AbstractAscendMemoryPoolSupport
  void ResetIdleMemBuf() const override { instance_->ResetIdleMemBuf(); }

  bool RecordEvent(int64_t task_id_on_stream, uint32_t user_stream_id,
                   const std::vector<std::pair<uint32_t, DeviceMemPtr>> &memory_stream_addresses,
                   const DeviceEventPtr &event) override {
    return instance_->RecordEvent(task_id_on_stream, user_stream_id, memory_stream_addresses, event);
  }

  bool WaitEvent(int64_t task_id_on_stream, uint32_t user_stream_id, uint32_t memory_stream_id) override;

  bool WaitEvent(int64_t task_id_on_stream, uint32_t memory_stream_id) override;

  bool SyncAllEvents() override;

  size_t AlignMemorySize(size_t size) const override { return instance_->AlignMemorySize(size); }

  size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) override {
    return instance_->CalMemBlockAllocSize(size, from_persistent_mem, need_recycle);
  }

  void SetMemPoolBlockSize(size_t available_device_mem_size) override {
    instance_->SetMemPoolBlockSize(available_device_mem_size);
  }

  size_t MemAllocUnitSize(bool from_persistent_mem) const override {
    return instance_->MemAllocUnitSize(from_persistent_mem);
  }

  void SetMemAllocUintSize(size_t common_size, size_t persist_size = kDynamicMemAllocUnitSize) override {
    instance_->SetMemAllocUintSize(common_size, persist_size);
  }

  void *GetMinUsingMemoryAddr() const override { return instance_->GetMinUsingMemoryAddr(); }

  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override { return instance_->AllocDeviceMem(size, addr); }

  bool FreeDeviceMem(const DeviceMemPtr &addr) override { return instance_->FreeDeviceMem(addr); }

  size_t free_mem_size() override { return instance_->free_mem_size(); }

  uint64_t total_mem_size() const override { return instance_->total_mem_size(); }

  size_t GetMaxUsedMemSize() const override { return instance_->GetMaxUsedMemSize(); }

  size_t GetVmmUsedMemSize() const override { return instance_->GetVmmUsedMemSize(); }

  void DumpDynamicMemPoolDebugInfo() override { instance_->DumpDynamicMemPoolDebugInfo(); }

  size_t TotalMemStatistics() const override { return instance_->TotalMemStatistics(); }

  size_t TotalUsedMemStatistics() const override { return instance_->TotalUsedMemStatistics(); }

  size_t TotalUsedByEventMemStatistics() const override { return instance_->TotalUsedByEventMemStatistics(); }

  size_t TotalIdleMemStatistics() const override { return instance_->TotalIdleMemStatistics(); }

  size_t TotalEagerFreeMemStatistics() const override { return instance_->TotalEagerFreeMemStatistics(); }

  size_t UsedMemPeakStatistics() const override { return instance_->UsedMemPeakStatistics(); }

  size_t MaxMemAllocatedStatistics() const override { return instance_->MaxMemAllocatedStatistics(); }

  size_t MaxMemReservedStatistics() const override { return instance_->MaxMemReservedStatistics(); }

  size_t ActualPeakStatistics() const override { return instance_->ActualPeakStatistics(); }

  std::unordered_map<std::string, std::size_t> BlockCountsStatistics() const override {
    return std::move(instance_->BlockCountsStatistics());
  }

  std::unordered_map<std::string, std::size_t> BlockUnitSizeStatistics() const override {
    return std::move(instance_->BlockUnitSizeStatistics());
  }

  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> CommonMemBlocksInfoStatistics()
    const override {
    return std::move(instance_->CommonMemBlocksInfoStatistics());
  }

  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> PersistentMemBlocksInfoStatistics()
    const override {
    return std::move(instance_->PersistentMemBlocksInfoStatistics());
  }

  void ResetMaxMemReserved() override { instance_->ResetMaxMemReserved(); }

  void ResetMaxMemAllocated() override { instance_->ResetMaxMemAllocated(); }

  const bool IsEnableEagerFree() const override { return instance_->IsEnableEagerFree(); }

  const bool IsEnableVmm() const override { return instance_->IsEnableVmm(); }

  void SetEnableVmm(bool enable_vmm) override { instance_->SetEnableVmm(enable_vmm); }

  const bool SyncAllStreams() override { return instance_->SyncAllStreams(); }

  size_t AllocDeviceMemByEagerFree(size_t size, DeviceMemPtr *addr) override {
    return instance_->AllocDeviceMemByEagerFree(size, addr);
  }

  size_t FreeDeviceMemByEagerFree(const DeviceMemPtr addr, const size_t size) override {
    return instance_->FreeDeviceMemByEagerFree(addr, size);
  }

  size_t MmapDeviceMem(size_t size, DeviceMemPtr addr) override { return instance_->MmapDeviceMem(size, addr); }

  std::string GetMemoryPoolType() const override { return "DefaultEnhancedAscendMemoryPool"; }

  void ReportMemoryPoolInfo() override { instance_->ReportMemoryPoolInfo(); }

  bool IsEnableTimeEvent() override { return instance_->IsEnableTimeEvent(); }

  void SetEnableTimeEvent(bool enable_time_event) override { instance_->SetEnableTimeEvent(enable_time_event); }

  MemoryTimeEventPtr GenAllocateMemoryTimeEvent(const void *addr, size_t size, uint32_t stream_id, bool from_persistent,
                                                bool is_persistent) override {
    return instance_->GenAllocateMemoryTimeEvent(addr, size, stream_id, from_persistent, is_persistent);
  }

  MemoryTimeEventPtr GenFreeMemoryTimeEvent(const void *addr) override {
    return instance_->GenFreeMemoryTimeEvent(addr);
  }

 private:
  DefaultAscendMemoryPoolPtr instance_;
  size_t last_vmm_used_size_{0};
};

class BACKEND_EXPORT BestFitAscendMemoryPool : public AbstractAscendMemoryPoolSupport, public DynamicMemPoolBestFit {
 public:
  BestFitAscendMemoryPool();
  BestFitAscendMemoryPool(const BestFitAscendMemoryPool &) = delete;
  BestFitAscendMemoryPool &operator=(const BestFitAscendMemoryPool &) = delete;
  ~BestFitAscendMemoryPool() override = default;

  void SetMemPoolBlockSize(size_t available_device_mem_size) override {
    return AbstractAscendMemoryPoolSupport::SetMemPoolBlockSize(available_device_mem_size);
  }

  size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) override {
    return AbstractAscendMemoryPoolSupport::CalMemBlockAllocSize(size, from_persistent_mem, need_recycle);
  }

  const bool IsEnableEagerFree() const override { return AbstractAscendMemoryPoolSupport::IsEnableEagerFree(); }

  std::string GetMemoryPoolType() const override { return "BestFitAscendMemoryPool"; }

  void ReportMemoryTimeEvent(const MemoryTimeEventPtr &time_event) override;
};

class BACKEND_EXPORT AscendMemoryPool {
 public:
  AscendMemoryPool(const AscendMemoryPool &) = delete;
  AscendMemoryPool &operator=(const AscendMemoryPool &) = delete;

  static AbstractAscendMemoryPoolSupport &GetInstance() {
    static std::once_flag flag;
    std::call_once(flag, [&]() {
      if (UseOldMemoryPool()) {
        instance_ = std::make_shared<BestFitAscendMemoryPool>();
        enhanced_instance_ = instance_;
      } else {
        auto pool = std::make_shared<DefaultAscendMemoryPool>();
        instance_ = pool;
        enhanced_instance_ = std::make_shared<DefaultEnhancedAscendMemoryPool>(pool);
        if (UseEnhancedMemoryPool()) {
          instance_ = enhanced_instance_;
        }
      }
      // Initialize instance and set ptr.
      float init_size = runtime::RuntimeConf::GetInstance()->mem_init_size();
      size_t init_size_byte = FloatToSize(init_size * kGBToByte);
      float increase_size = runtime::RuntimeConf::GetInstance()->mem_block_increase_size();
      size_t increase_size_byte = FloatToSize(increase_size * kGBToByte);
      float max_size = runtime::RuntimeConf::GetInstance()->mem_max_size();
      size_t max_size_byte = FloatToSize(max_size * kGBToByte);
      instance_->Initialize(init_size_byte, increase_size_byte, max_size_byte);
      pool_ = instance_;
    });
    return *pool_;
  }

  static void SetEnhancedMemoryPool(bool enable) { pool_ = enable ? enhanced_instance_ : instance_; }

 private:
  AscendMemoryPool() {}

  static bool UseOldMemoryPool() {
    if (common::IsDisableAllocConfig(common::kAllocMemoryPool)) {
      return false;
    }
    return IsDisableGeKernel() || common::IsEnableAllocConfig(common::kAllocMemoryPool);
  }

  // Use enhanced memory pool when enable debug, enable log, enable prof, dry run and so on.
  static bool UseEnhancedMemoryPool() {
    bool enable_debugger = false;
#ifdef ENABLE_DEBUGGER
    auto profiler = profiler::Profiler::GetInstance(kCPUDevice);
    if (profiler != nullptr && profiler->GetEnableFlag() && profiler->GetProfileMemoryFlag()) {
      enable_debugger = true;
    }
#endif
    auto submodule = common::GetEnv("MS_SUBMODULE_LOG_v");
    bool enable_pre_act_log = ParseDebugConfig(submodule, "PRE_ACT") == "0";
    bool enable_debug_log = common::GetEnv("GLOG_v") == "0";
    return enable_debugger || enable_pre_act_log || enable_debug_log ||
           MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_PROF_MEM) ||
           common::IsEnableAllocConfig(common::kAllocMemoryTracker) ||
           common::IsEnableRuntimeConfig(common::kRuntimeMemoryStat) || common::IsDryRun();
  }

  static std::string ParseDebugConfig(std::string input, std::string config) {
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

  // Reference to memory pool.
  static AbstractAscendMemoryPoolSupportPtr pool_;

  // Basic memory pool instance with high performance.
  static AbstractAscendMemoryPoolSupportPtr instance_;

  // Memory pool support profiling and debugging.
  static AbstractAscendMemoryPoolSupportPtr enhanced_instance_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_POOL_H_
