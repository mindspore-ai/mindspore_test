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

#include "backend/ge_backend/executor/ge_memory_manager.h"
#include <algorithm>
#include "include/runtime/memory/mem_pool/mem_tracker.h"
#include "include/runtime/utils/runtime_conf/runtime_env.h"

namespace mindspore {
namespace backend {
namespace ge_backend {
GEMemoryManager &GEMemoryManager::Instance() {
  static GEMemoryManager instance{};
  return instance;
}

void GEMemoryManager::InitGEMemory(const backend::ge_backend::RunOptions &run_options, size_t workspace_memory_size,
                                   size_t fixed_memory_size, size_t const_memory_size, bool is_refreshable,
                                   size_t stream_id) {
  auto graph_name = run_options.name;
  if (graph_memory_.find(graph_name) != graph_memory_.end()) {
    MS_LOG(EXCEPTION) << "Graph " << graph_name << " has been initialized.";
  }
  MS_LOG(INFO) << "GE graph name: " << graph_name << ", workspace memory size: " << workspace_memory_size
               << ", fixed memory size: " << fixed_memory_size << ", const memory size: " << const_memory_size
               << ", stream id: " << stream_id;
  if (runtime::IsEnableRuntimeConfig(runtime::kRuntimeMemoryStat)) {
    std::cout << "[MS_RUNTIME_PROF] GE graph name: " << graph_name
              << ", workspace memory size: " << workspace_memory_size << ", fixed memory size: " << fixed_memory_size
              << ", const memory size: " << const_memory_size << ", stream id: " << stream_id << std::endl;
  }
  GEMemory ge_memory;
  ge_memory.run_options = run_options;
  ge_memory.workspace_memory = workspace_memory_size;
  ge_memory.fixed_memory = fixed_memory_size;
  ge_memory.const_memory = const_memory_size;
  ge_memory.is_refreshable = is_refreshable;
  ge_memory.stream_id = stream_id;
  graph_memory_[graph_name] = ge_memory;
  stream_id_to_graphs_[stream_id].insert(graph_name);
  auto iter = stream_id_to_fix_memory_.find(stream_id);
  if (iter == stream_id_to_fix_memory_.end() || iter->second == nullptr || iter->second->has_alloc) {
    stream_id_to_fix_memory_[stream_id] = std::make_shared<FixedMemory>();
  }
  auto fixed_memory = stream_id_to_fix_memory_[stream_id];
  graph_memory_[graph_name].reuse_memory = fixed_memory;
  fixed_memory->memory_size = std::max(fixed_memory->memory_size, fixed_memory_size);
}

std::set<FixedMemoryPtr> GEMemoryManager::GetAllNotAllocFixMemory() const {
  std::set<FixedMemoryPtr> fix_memory_ptrs;
  for (auto &kv : graph_memory_) {
    if (!kv.second.reuse_memory->has_alloc) {
      fix_memory_ptrs.insert(kv.second.reuse_memory);
    }
  }
  return fix_memory_ptrs;
}

void GEMemoryManager::AllocGEMemory(GEAllocFunc alloc_func, GEUpdateMemoryFunc update_func) const {
  auto need_alloc_ptrs = GetAllNotAllocFixMemory();
  if (need_alloc_ptrs.empty()) {
    return;
  }
  for (auto ptr : need_alloc_ptrs) {
    ptr->memory_ptr = alloc_func(ptr->memory_size);
    if (ptr->memory_ptr == nullptr) {
      MS_LOG(EXCEPTION) << "Out of memory, Alloc GE memory failed, size: " << ptr->memory_size;
    }
    ptr->has_alloc = true;
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddTask, "AllocGEMemory", "", "", false);
    device::tracker::CALL_MEMORY_TRACKER_WITH_FILE(AddCompileTimeMemInfo, "AllocGEMemory", ptr->memory_size,
                                                   ptr->memory_ptr, memory::mem_pool::MemType::kGeFixed);
  }
  for (auto &kv : graph_memory_) {
    if (need_alloc_ptrs.find(kv.second.reuse_memory) == need_alloc_ptrs.end()) {
      continue;
    }
    MS_EXCEPTION_IF_NULL(kv.second.reuse_memory->memory_ptr);
    MS_EXCEPTION_IF_CHECK_FAIL(kv.second.reuse_memory->memory_size >= kv.second.fixed_memory,
                               "Fixed memory size error, graph name: " + kv.first);
    update_func(kv.second.is_refreshable, kv.second.run_options, kv.second.reuse_memory->memory_ptr,
                kv.second.fixed_memory);
  }
}

size_t GEMemoryManager::GetWorkspaceMemory(const std::string &graph_name) const {
  auto iter = graph_memory_.find(graph_name);
  if (iter == graph_memory_.end()) {
    MS_LOG(EXCEPTION) << "Graph " << graph_name << " has not been initialized.";
  }
  return iter->second.workspace_memory;
}

void GEMemoryManager::Clear() {
  graph_memory_.clear();
  stream_id_to_graphs_.clear();
  stream_id_to_fix_memory_.clear();
}
}  // namespace ge_backend
}  // namespace backend
}  // namespace mindspore
