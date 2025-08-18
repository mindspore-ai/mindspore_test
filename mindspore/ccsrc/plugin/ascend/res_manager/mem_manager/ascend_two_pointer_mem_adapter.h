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

#ifndef MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_TWO_POINTER_MEM_ADAPTER_H_
#define MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_TWO_POINTER_MEM_ADAPTER_H_

#include "plugin/ascend/res_manager/mem_manager/ascend_memory_adapter.h"
#include <string>
#include <memory>
#include <vector>

namespace mindspore {
namespace device {
namespace ascend {
class AscendTwoPointerMemAdapter : public AscendMemAdapter {
 public:
  bool Initialize() override;
  bool DeInitialize() override;
  uint8_t *MallocStaticDevMem(size_t size, const std::string &tag = "") override;
  bool FreeStaticDevMem(void *addr) override;
  uint8_t *MallocDynamicDevMem(size_t size, const std::string &tag = "") override;
  void ResetDynamicMemory() override;
  void SimulationInitialize() override;
  std::string DevMemStatistics() const override;
  size_t GetDynamicMemUpperBound(void *min_static_addr) const override;
  [[nodiscard]] uint64_t FreeDevMemSize() const override;

 private:
  std::string DevMemDetailInfo() const;
  uint8_t *device_mem_base_addr_{nullptr};
  // static memory info, from a high address to a low address
  int64_t static_mem_offset_{0};
  // dynamic memory info, from a low address to a high address
  int64_t cur_dynamic_mem_offset_{0};
  // Maximum dynamic memory have already allocated, dynamically updated
  int64_t max_dynamic_mem_offset_{0};
  // History maximum dynamic memory (used in memory pool recycle mode)
  int64_t history_max_dynamic_mem_offset_{0};
  std::vector<std::shared_ptr<MemoryBlock>> dynamic_memory_block_list_;
  std::vector<std::shared_ptr<MemoryBlock>> static_memory_block_list_;
};
}  // namespace ascend
}  // namespace device
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_DEVICE_ASCEND_ASCEND_TWO_POINTER_MEM_ADAPTER_H_
