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

#include "include/backend/mem_reuse/abstract_dynamic_mem_pool.h"
#include "include/backend/mem_reuse/enhanced_dynamic_mem_pool.h"
#include "include/backend/mem_reuse/mem_dynamic_allocator.h"
#include "include/backend/visible.h"
#include "plugin/device/ascend/hal/device/abstract_ascend_memory_pool_support.h"
#include "plugin/device/ascend/hal/profiler/ascend_profiling.h"
#include "utils/hash_map.h"
#include "utils/ms_context.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
class BACKEND_EXPORT DefaultAscendMemoryPool : public AbstractAscendMemoryPoolSupport, public AbstractDynamicMemPool {
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

class BACKEND_EXPORT DefaultEnhancedAscendMemoryPool : public AbstractAscendMemoryPoolSupport,
                                                       public EnhancedDynamicMemPool {
 public:
  DefaultEnhancedAscendMemoryPool();
  DefaultEnhancedAscendMemoryPool(const DefaultEnhancedAscendMemoryPool &) = delete;
  DefaultEnhancedAscendMemoryPool &operator=(const DefaultEnhancedAscendMemoryPool &) = delete;
  ~DefaultEnhancedAscendMemoryPool() override = default;

  std::string GetMemoryPoolType() const override { return "DefaultEnhancedAscendMemoryPool"; }

  void SetMemPoolBlockSize(size_t available_device_mem_size) override {
    return AbstractAscendMemoryPoolSupport::SetMemPoolBlockSize(available_device_mem_size);
  }

  size_t CalMemBlockAllocSize(size_t size, bool from_persistent_mem, bool need_recycle = false) override {
    return AbstractAscendMemoryPoolSupport::CalMemBlockAllocSize(size, from_persistent_mem, need_recycle);
  }

  const bool IsEnableEagerFree() const override { return AbstractAscendMemoryPoolSupport::IsEnableEagerFree(); }
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
      } else {
        if (UseEnhancedMemoryPool()) {
          instance_ = std::make_shared<DefaultEnhancedAscendMemoryPool>();
        } else {
          instance_ = std::make_shared<DefaultAscendMemoryPool>();
        }
      }
    });
    return *instance_;
  }

  static bool UseOldMemoryPool() {
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
           common::IsEnableRuntimeConfig(common::kRuntimeMemoryStat) || common::IsNeedProfileMemory();
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

 private:
  static std::shared_ptr<AbstractAscendMemoryPoolSupport> instance_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_POOL_H_
