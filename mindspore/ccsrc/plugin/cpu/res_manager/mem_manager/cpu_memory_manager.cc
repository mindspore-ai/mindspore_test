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

#include "plugin/cpu/res_manager/mem_manager/cpu_memory_manager.h"
#include "include/utils/anfalgo.h"
#include "utils/ms_context.h"
#include "include/utils/convert_utils.h"
namespace mindspore {
namespace device {
namespace cpu {
uint8_t *CPUMemoryManager::MemMalloc(size_t size) {
  auto block = std::make_shared<std::vector<uint8_t>>();
  try {
    block->resize(size, 0);
    auto ptr = block->data();
    mem_block_map_[ptr] = block;
    return ptr;
  } catch (const std::exception &e) {
    MS_LOG(EXCEPTION) << "Malloc memory failed: size " << size;
  }
}

uint8_t *CPUMemoryManager::MallocStaticMem(size_t size, bool, uint32_t) {
  auto ptr = MemMalloc(size);
  static_mem_[ptr] = size;
  return ptr;
}

uint8_t *CPUMemoryManager::MallocDynamicMem(size_t size, bool) {
  void *ptr = nullptr;
  size_t min_size = 0;
  // first find the smallest cached_mem_ which fits the size
  for (auto &&iter : cached_mem_) {
    if (iter.second >= size) {
      if (min_size == 0 || iter.second < min_size) {
        ptr = iter.first;
        min_size = iter.second;
      }
    }
  }
  if (ptr != nullptr) {
    if (memset_s(ptr, size, 0, size) != EOK) {
      free(ptr);
      MS_LOG(EXCEPTION) << "Failed to init memory.";
    }
    dynamic_mem_[ptr] = min_size;
    (void)cached_mem_.erase(ptr);
    return reinterpret_cast<uint8_t *>(ptr);
  }
  // if not found, malloc
  auto new_ptr = MemMalloc(size);
  dynamic_mem_[new_ptr] = size;
  return new_ptr;
}

void CPUMemoryManager::ResetDynamicMemory() {
  // don't free, for multi graph
  for (auto &&iter : dynamic_mem_) {
    cached_mem_[iter.first] = iter.second;
  }
  dynamic_mem_.clear();
}

CPUMemoryManager::~CPUMemoryManager() { MemFree(); }

void CPUMemoryManager::MemFree() noexcept {
  if (mem_ptr_ != nullptr) {
    mem_ptr_ = nullptr;
    mem_size_ = 0;
  }
  static_mem_.clear();
  dynamic_mem_.clear();
  cached_mem_.clear();
  mem_block_map_.clear();
}

void *CPUMemoryManager::StaticMemMalloc(size_t mem_size) {
  auto ptr = MemMalloc(mem_size);
  if (ptr != nullptr) {
    static_mem_[ptr] = mem_size;
    return ptr;
  } else {
    MS_LOG(EXCEPTION) << "Malloc memory failed: size " << mem_size;
  }
}

void CPUMemoryManager::MemFree(void *ptr) {
  auto iter = static_mem_.find(ptr);
  if (iter != static_mem_.end()) {
    (void)static_mem_.erase(iter);
    auto block_iter = mem_block_map_.find(ptr);
    if (block_iter != mem_block_map_.end()) {
      (void)mem_block_map_.erase(block_iter);
    }
  }
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
