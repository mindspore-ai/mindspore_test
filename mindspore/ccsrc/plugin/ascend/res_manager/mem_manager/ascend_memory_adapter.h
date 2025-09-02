/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_ADAPTER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_ADAPTER_H_

#include <algorithm>
#include <mutex>
#include <string>
#include <memory>
#include <vector>
#include <limits>

#include "ir/anf.h"
#include "plugin/ascend/res_manager/visible.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
namespace ascend {
struct MemoryBlock {
  MemoryBlock(void *ptr, const size_t size, const std::string &tag) {
    mem_ptr = ptr;
    mem_size = size;
    mem_tag = tag;
  }

  void *mem_ptr{nullptr};
  size_t mem_size{0};
  std::string mem_tag;
};

class AscendMemAdapter;
using AscendMemAdapterPtr = std::shared_ptr<AscendMemAdapter>;

class ASCEND_RES_MANAGER_EXPORT AscendMemAdapter {
 public:
  virtual ~AscendMemAdapter() = default;
  static AscendMemAdapterPtr GetInstance();

  virtual bool Initialize();
  virtual bool DeInitialize();

  virtual uint8_t *MallocStaticDevMem(size_t size, const std::string &tag = "") = 0;
  virtual bool FreeStaticDevMem(void *addr) = 0;
  virtual uint8_t *MallocDynamicDevMem(size_t size, const std::string &tag = "") = 0;
  virtual void ResetDynamicMemory() = 0;
  virtual std::string DevMemStatistics() const = 0;
  virtual size_t GetDynamicMemUpperBound(void *min_static_addr) const = 0;
  [[nodiscard]] virtual uint64_t FreeDevMemSize() const = 0;

  virtual void SimulationInitialize();

  int64_t GetActualPeakMemory() const { return actual_peak_memory_; }
  int64_t GetUsedPeakMemory() const { return used_peak_memory_; }
  void UpdateActualPeakMemory(int64_t memory) { actual_peak_memory_ = std::max(actual_peak_memory_, memory); }
  void UpdateUsedPeakMemory(int64_t memory) { used_peak_memory_ = std::max(used_peak_memory_, memory); }
  [[nodiscard]] uint64_t MaxHbmSizeForMs() const { return max_available_ms_hbm_size_; }
  [[nodiscard]] int64_t GetMsUsedHbmSize() const { return ms_used_hbm_size_; }
  static size_t GetRoundUpAlignSize(size_t input_size);
  static size_t GetRoundDownAlignSize(size_t input_size);
  uint8_t *MallocAlign32FromRts(size_t size) const;
  bool FreeAlign32ToRts(void *devPtr) const;

 protected:
  AscendMemAdapter() = default;
  uint8_t *MallocFromRts(size_t size) const;
  bool FreeToRts(void *devPtr, const size_t size) const;

  bool initialized_{false};
  // Support multi-thread.
  std::mutex mutex_;

  // Actual peak memory usage (with fragments)
  int64_t actual_peak_memory_{0};
  // Used peak memory usage (without fragments)
  int64_t used_peak_memory_{0};

  // rts Memory INFO
  size_t device_hbm_total_size_{0};
  size_t device_hbm_free_size_{0};
  size_t device_hbm_huge_page_reserved_size_{0};

  int64_t ms_used_hbm_size_{0};
  int64_t max_available_ms_hbm_size_{0};

 private:
  DISABLE_COPY_AND_ASSIGN(AscendMemAdapter)
  size_t GetDeviceMemSizeFromContext() const;
  static AscendMemAdapterPtr instance_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_MEMORY_ADAPTER_H_
