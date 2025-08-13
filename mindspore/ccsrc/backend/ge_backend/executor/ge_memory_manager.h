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
#ifndef MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_MEMORY_MANAGER_H_
#define MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_MEMORY_MANAGER_H_

#include <memory>
#include <string>
#include <set>
#include <map>
#include "backend/ge_backend/graph_ir/types.h"
#include "utils/ms_utils.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
struct FixedMemory {
  size_t memory_size = 0;
  bool has_alloc = false;
  void *memory_ptr;
};
using FixedMemoryPtr = std::shared_ptr<FixedMemory>;
using GEAllocFunc = std::function<void *(size_t)>;
using GEUpdateMemoryFunc = std::function<backend::ge_backend::Status(
  bool is_refreshable, const backend::ge_backend::RunOptions &options, const void *const memory, size_t size)>;

struct GEMemory {
  backend::ge_backend::RunOptions run_options;
  size_t workspace_memory;
  size_t fixed_memory;
  size_t const_memory;
  bool is_refreshable;
  size_t stream_id;
  // Let different GE graphs share a fix memory
  FixedMemoryPtr reuse_memory;
};

class GEMemoryManager {
 public:
  static GEMemoryManager &Instance();
  void InitGEMemory(const backend::ge_backend::RunOptions &run_options, size_t workspace_memory_size,
                    size_t fixed_memory_size, size_t const_memory_size, bool is_refreshable, size_t stream_id);
  void AllocGEMemory(GEAllocFunc alloc_func, GEUpdateMemoryFunc update_func) const;
  size_t GetWorkspaceMemory(const std::string &graph_name) const;
  void Clear();

 private:
  GEMemoryManager() = default;
  ~GEMemoryManager() = default;
  DISABLE_COPY_AND_ASSIGN(GEMemoryManager);
  std::set<FixedMemoryPtr> GetAllNotAllocFixMemory() const;
  std::map<std::string, GEMemory> graph_memory_;
  std::map<size_t, std::set<std::string>> stream_id_to_graphs_;
  std::map<size_t, FixedMemoryPtr> stream_id_to_fix_memory_;
};
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_GE_BACKEND_EXECUTOR_GE_MEMORY_MANAGER_H_
