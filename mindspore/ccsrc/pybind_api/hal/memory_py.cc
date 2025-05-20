/**
 * Copyright 2024-2025 Huawei Technologies Co., Ltd
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

#include "pybind_api/hal/memory_py.h"
#include "runtime/pipeline/pipeline.h"
#include "runtime/hardware/device_context.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/device/res_manager/hal_res_manager.h"

namespace mindspore {
namespace hal {
namespace {
device::HalResBase *GetResManager() {
  auto ms_context = MsContext::GetInstance();
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::ResKey res_key{device::GetDeviceTypeByName(device_name), device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  if (!res_manager) {
    MS_LOG(WARNING) << "Device  " << device_name << " is not created yet.";
  }
  return res_manager;
}

py::dict CreateEmptyMemoryStats() {
  py::dict memory_stats;
  py::dict commom_mem_pool_stats;
  py::dict persistent_mem_pool_stats;
  memory_stats["total_reserved_memory"] = 0;
  memory_stats["total_allocated_memory"] = 0;
  memory_stats["total_idle_memory"] = 0;
  memory_stats["total_eager_free_memory"] = 0;
  memory_stats["max_reserved_memory"] = 0;
  memory_stats["max_allocated_memory"] = 0;
  commom_mem_pool_stats["block_unit_size"] = 0;
  commom_mem_pool_stats["block_counts"] = 0;
  commom_mem_pool_stats["blocks_info"] =
    std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>{};
  persistent_mem_pool_stats["block_counts"] = 0;
  persistent_mem_pool_stats["block_unit_size"] = 0;
  persistent_mem_pool_stats["blocks_info"] =
    std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>>{};
  memory_stats["commom_mem_pool_stats"] = commom_mem_pool_stats;
  memory_stats["persistent_mem_pool_stats"] = persistent_mem_pool_stats;
  return memory_stats;
}
}  // namespace

py::dict MemoryStats(const std::string &device_target) {
  runtime::Pipeline::Get().WaitAll();
  auto res_manager = GetResManager();
  if (res_manager == nullptr) {
    return CreateEmptyMemoryStats();
  }

  // Memory statistics result to be returned.
  py::dict memory_stats;
  py::dict commom_mem_pool_stats;
  py::dict persistent_mem_pool_stats;
  // Peak memory statistics.
  // py::dict peak_mem_stats;

  size_t total_mem_size = res_manager->GetTotalMemStatistics();
  size_t total_used_mem_size = res_manager->GetTotalUsedMemStatistics();
  size_t total_idle_mem_size = res_manager->GetTotalIdleMemStatistics();
  size_t total_eager_free_mem_size = res_manager->GetTotalEagerFreeMemStatistics();
  size_t used_mem_peak_size = res_manager->GetUsedMemPeakStatistics();
  size_t reserved_mem_peak_size = res_manager->GetReservedMemPeakStatistics();
  std::unordered_map<std::string, std::size_t> block_counts_stats = res_manager->GetBlockCountsStatistics();
  std::unordered_map<std::string, std::size_t> block_unit_size_stats = res_manager->GetBlockUnitSizeStatistics();
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> common_mem_blocks_info =
    res_manager->GetCommonMemBlocksInfoStatistics();
  std::unordered_map<device::DeviceMemPtr, std::unordered_map<std::string, size_t>> persistent_mem_blocks_info =
    res_manager->GetPersistentMemBlocksInfoStatistics();

  memory_stats["total_reserved_memory"] = total_mem_size;
  memory_stats["total_allocated_memory"] = total_used_mem_size;
  memory_stats["total_idle_memory"] = total_idle_mem_size;
  memory_stats["total_eager_free_memory"] = total_eager_free_mem_size;
  memory_stats["max_reserved_memory"] = reserved_mem_peak_size;
  memory_stats["max_allocated_memory"] = used_mem_peak_size;
  commom_mem_pool_stats["block_unit_size"] = block_unit_size_stats["common_mem_pool"];
  commom_mem_pool_stats["block_counts"] = block_counts_stats["common_mem_pool"];
  commom_mem_pool_stats["blocks_info"] = common_mem_blocks_info;
  persistent_mem_pool_stats["block_counts"] = block_counts_stats["persistent_mem_pool"];
  persistent_mem_pool_stats["block_unit_size"] = block_unit_size_stats["persistent_mem_pool"];
  persistent_mem_pool_stats["blocks_info"] = persistent_mem_blocks_info;
  memory_stats["commom_mem_pool_stats"] = commom_mem_pool_stats;
  memory_stats["persistent_mem_pool_stats"] = persistent_mem_pool_stats;
  return memory_stats;
}

void ResetMaxMemoryReserved(const std::string &device_target) {
  runtime::Pipeline::Get().WaitAll();
  auto res_manager = GetResManager();
  if (res_manager == nullptr) {
    return;
  }

  res_manager->ResetMaxMemoryReserved();
}

void ResetMaxMemoryAllocated(const std::string &device_target) {
  runtime::Pipeline::Get().WaitAll();
  auto res_manager = GetResManager();
  if (res_manager == nullptr) {
    return;
  }

  res_manager->ResetMaxMemoryAllocated();
}

size_t EmptyCache(const std::string &device_target) {
  runtime::Pipeline::Get().WaitAll();
  auto res_manager = GetResManager();
  if (res_manager == nullptr) {
    return -1L;
  }

  return res_manager->EmptyCache();
}

void RegMemory(py::module *m) {
  (void)m->def("_memory_stats", &mindspore::hal::MemoryStats, "Get memory pool's statistics.");
  (void)m->def("_reset_max_mem_reserved", &mindspore::hal::ResetMaxMemoryReserved,
               "Reset the maximum recorded memory reserved.");
  (void)m->def("_reset_max_mem_allocated", &mindspore::hal::ResetMaxMemoryAllocated,
               "Reset the maximum recorded memory allocated.");
  (void)m->def("_empty_cache", &mindspore::hal::EmptyCache, "Empty memory pool cache.");
}
}  // namespace hal
}  // namespace mindspore
