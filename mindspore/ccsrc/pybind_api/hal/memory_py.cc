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
#include <fstream>
#include <vector>
#include <map>
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

namespace {
std::vector<std::string> Split(const std::string &s, const std::string &delimiter) {
  size_t pos_start = 0;
  size_t pos_end;
  size_t delim_len = delimiter.length();
  std::string token;
  std::vector<std::string> res;

  while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
    token = s.substr(pos_start, pos_end - pos_start);
    pos_start = pos_end + delim_len;
    res.push_back(token);
  }

  res.push_back(s.substr(pos_start));
  return res;
}

template <typename T>
T Parse(const std::string &s) {
  std::stringstream sstream(s);
  T ans;
  sstream >> ans;
  return ans;
}

struct MemoryBlock {
  static constexpr size_t kMemBlockFileSizeMinSize = 10;
  static constexpr size_t kStartTimeStampIdx = 0;
  static constexpr size_t kEndTimeStampIdx = 1;
  static constexpr size_t kStreamIdIdx = 3;
  static constexpr size_t kSizeIdx = 5;
  static constexpr size_t kActualPeakMemIdx = 6;
  static constexpr size_t kTypeIdx = 9;
  static constexpr size_t kInvalidValue = 0;

  explicit MemoryBlock(const std::string &block_string) {
    auto &&elements = Split(block_string, ",");
    MS_EXCEPTION_IF_CHECK_FAIL(elements.size() > kMemBlockFileSizeMinSize, "Invalid line : " + block_string);
    start_time_stamp_ = Parse<size_t>(elements[kStartTimeStampIdx]);
    MS_EXCEPTION_IF_CHECK_FAIL(start_time_stamp_ != kInvalidValue,
                               "Invalid start_time_stamp_: " + elements[kStartTimeStampIdx]);
    end_time_stamp_ = Parse<size_t>(elements[kEndTimeStampIdx]);
    MS_EXCEPTION_IF_CHECK_FAIL(end_time_stamp_ != kInvalidValue,
                               "Invalid end_time_stamp_: " + elements[kEndTimeStampIdx]);
    stream_id_ = Parse<uint32_t>(elements[kStreamIdIdx]);
    size_ = Parse<size_t>(elements[kSizeIdx]);
    MS_EXCEPTION_IF_CHECK_FAIL(size_ != kInvalidValue, "Invalid size_: " + elements[kSizeIdx]);
    actual_peak_mem_ = Parse<size_t>(elements[kActualPeakMemIdx]);
    MS_EXCEPTION_IF_CHECK_FAIL(actual_peak_mem_ != kInvalidValue,
                               "Invalid actual_peak_mem_: " + elements[kActualPeakMemIdx]);
    type_ = Parse<std::string>(elements[kIndex9]);
  }

  size_t start_time_stamp_;
  size_t end_time_stamp_;
  uint32_t stream_id_;
  size_t size_;
  size_t actual_peak_mem_;
  std::string type_;

  bool IsPersistent() { return type_ == "ConstantValue" || type_ == "Weight" || type_ == "GeConst"; }
};
}  // namespace

struct MemoryReplayProcesser {
  MemoryReplayProcesser() {
    auto ms_context = MsContext::GetInstance();
    auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
    const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
    device::ResKey res_key{device::GetDeviceTypeByName(device_name), device_id};
    res_manager_ = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  }

  ~MemoryReplayProcesser() = default;

  void operator()(const std::string &file_path) {
    MS_EXCEPTION_IF_NULL(res_manager_);
    res_manager_->Initialize();
    MS_EXCEPTION_IF_NULL(res_manager_->mem_manager());
    auto mem_pool = res_manager_->mem_manager()->GetMemoryPool();
    MS_EXCEPTION_IF_NULL(mem_pool);

    std::ifstream tracker_file(file_path, std::ios::in);
    if (!tracker_file.is_open()) {
      MS_LOG(EXCEPTION) << "Failed to open file: " << file_path << ". Please check whether the file exists.";
      return;
    }
    std::string line;
    size_t cur_time_stamp = 0L;
    size_t process_line_no = 0;
    while (std::getline(tracker_file, line)) {
      process_line_no++;
      // Skip title.
      if (process_line_no == 1) {
        continue;
      }
      MemoryBlock block(line);
      MS_EXCEPTION_IF_CHECK_FAIL(block.start_time_stamp_ >= cur_time_stamp,
                                 "Invalid memory block, line no : " + std::to_string(process_line_no));
      cur_time_stamp = block.start_time_stamp_;
      for (auto iter = to_free_mems_.begin(); iter != to_free_mems_.end();) {
        if (iter->first > block.start_time_stamp_) {
          break;
        }
        mem_pool->FreeTensorMem(iter->second);
        iter = to_free_mems_.erase(iter);
      }
      void *addr = mem_pool->AllocTensorMem(block.size_, block.IsPersistent(), false, block.stream_id_);
      // Record and compare peak value.
      size_t cur_peak = mem_pool->ActualPeakStatistics();
      if (block.actual_peak_mem_ != cur_peak) {
        MS_LOG(WARNING) << "Process line : " << process_line_no
                        << " block.actual_peak_mem_ : " << block.actual_peak_mem_
                        << " is not equal to cur peak : " << cur_peak << ".";
      }
      to_free_mems_[block.end_time_stamp_] = addr;
    }
    for (auto iter = to_free_mems_.begin(); iter != to_free_mems_.end();) {
      mem_pool->FreeTensorMem(iter->second);
      iter = to_free_mems_.erase(iter);
    }
  }

 private:
  device::HalResBase *res_manager_;
  std::map<size_t, void *> to_free_mems_;
};

void MemoryReplay(const std::string &file_path) {
  MemoryReplayProcesser memory_replay_processer;
  memory_replay_processer(file_path);
}

void RegMemory(py::module *m) {
  (void)m->def("_memory_stats", &mindspore::hal::MemoryStats, "Get memory pool's statistics.");
  (void)m->def("_reset_max_mem_reserved", &mindspore::hal::ResetMaxMemoryReserved,
               "Reset the maximum recorded memory reserved.");
  (void)m->def("_reset_max_mem_allocated", &mindspore::hal::ResetMaxMemoryAllocated,
               "Reset the maximum recorded memory allocated.");
  (void)m->def("_empty_cache", &mindspore::hal::EmptyCache, "Empty memory pool cache.");
  (void)m->def("_memory_replay", &mindspore::hal::MemoryReplay, py::arg("file_path"), "Memory replay.");
}
}  // namespace hal
}  // namespace mindspore
