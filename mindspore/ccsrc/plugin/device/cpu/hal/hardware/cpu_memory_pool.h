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

#ifndef MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_CPU_MEMORY_POOL_H_
#define MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_CPU_MEMORY_POOL_H_

#include <memory>
#include <mutex>
#include <string>

#include "include/backend/mem_reuse/mem_dynamic_allocator.h"
#include "runtime/runtime_conf/runtime_conf.h"
#include "utils/convert_utils_base.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace device {
namespace cpu {
class BACKEND_EXPORT CPUMemoryPool : public DynamicMemPoolBestFit {
 public:
  ~CPUMemoryPool() override = default;

  static CPUMemoryPool &GetInstance() {
    static CPUMemoryPool instance;
    static std::once_flag flag;
    std::call_once(flag, [&]() {
      float init_size = runtime::RuntimeConf::GetInstance()->mem_init_size();
      size_t init_size_byte = FloatToSize(init_size * kGBToByte);
      float increase_size = runtime::RuntimeConf::GetInstance()->mem_block_increase_size();
      size_t increase_size_byte = FloatToSize(increase_size * kGBToByte);
      float max_size = runtime::RuntimeConf::GetInstance()->mem_max_size();
      size_t max_size_byte = FloatToSize(max_size * kGBToByte);
      instance.Initialize(init_size_byte, increase_size_byte, max_size_byte);
    });
    return instance;
  }

  size_t AllocDeviceMem(size_t size, DeviceMemPtr *addr) override;
  bool FreeDeviceMem(const DeviceMemPtr &addr) override;
  size_t free_mem_size() override;
  std::string GetMemoryPoolType() const override { return "CPU"; }

 private:
  CPUMemoryPool() = default;
  DISABLE_COPY_AND_ASSIGN(CPUMemoryPool);

  size_t total_used_memory_{0};
};
}  // namespace cpu
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_HARDWARE_CPU_CPU_MEMORY_POOL_H_
