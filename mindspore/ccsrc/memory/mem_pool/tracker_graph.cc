/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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

#include "include/backend/mem_reuse/tracker_graph.h"

#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>

#include "include/backend/mem_reuse/race_checker.h"
#include "include/backend/mem_reuse/dynamic_mem_pool.h"
#include "ir/dtype.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/distributed_meta.h"
#include "include/common/debug/common.h"
#include "common/device_type.h"
#include "include/backend/mem_reuse/mem_dynamic_allocator.h"
#include "include/common/utils/utils.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace device {
namespace tracker {
namespace graph {
namespace {
std::string SerializeMap(const std::unordered_map<std::string, std::string> &map,
                         const std::string &pair_separator = ";", const std::string &key_value_separator = "=") {
  std::ostringstream oss;
  bool first = true;
  for (const auto &pair : map) {
    if (!first) {
      oss << pair_separator;
    }
    oss << pair.first << key_value_separator << pair.second;
    first = false;
  }
  return oss.str();
}

size_t GetTaskInfoStreamId(const TaskInfoPtr &task_info) {
  auto iter = task_info->attrs.find(kStreamId);
  if (iter == task_info->attrs.end()) {
    MS_LOG(ERROR) << "Stream id is not found , task info: " << task_info->time_stamp;
    return 0;
  }
  return std::stoul(iter->second);
}

constexpr size_t kLogThreshold = 10;
constexpr size_t kLogPersentage = 100;
}  // namespace

std::string TrackerTensor::ToString() { return "%" + std::to_string(start_time_stamp); }

std::string TrackerTensor::DtypeToString() { return TypeIdToString(dtype); }

std::string TrackerTensor::ShapeToString() { return "[" + ShapeVectorToString(shape) + "]"; }

std::string TrackerTensor::TensorInfoToString() {
  std::ostringstream oss;
  oss << DtypeToString() << ":" << ShapeToString();
  if (tensor_info != nullptr) {
    oss << "{";
    oss << "strdes=" << VectorToString(tensor_info->strides) << ",";
    oss << "offset=" << tensor_info->storage_offset;
    oss << "}";
  }
  return oss.str();
}

std::string TrackerOperator::name() {
  MS_EXCEPTION_IF_NULL(task_info);
  return task_info->node_name;
}

std::string TrackerOperator::ToString() {
  MS_EXCEPTION_IF_NULL(task_info);
  std::ostringstream oss;

  oss << "(";
  for (size_t i = 0; i < outputs.size(); i++) {
    oss << outputs[i]->ToString();
    if (i != outputs.size() - 1) {
      oss << ", ";
    }
  }
  oss << ") = " << task_info->node_name << "(";
  for (size_t i = 0; i < inputs.size(); i++) {
    oss << inputs[i]->ToString();
    if (i != inputs.size() - 1) {
      oss << ", ";
    }
  }
  oss << "), task_info: " << task_info->time_stamp << ", ";
  oss << "attrs {" << SerializeMap(task_info->attrs) << "} \n";

  oss << "    (";
  for (size_t i = 0; i < outputs.size(); i++) {
    oss << outputs[i]->TensorInfoToString();
    if (i != outputs.size() - 1) {
      oss << ", ";
    }
  }
  oss << ") <- (";
  for (size_t i = 0; i < inputs.size(); i++) {
    oss << inputs[i]->TensorInfoToString();
    if (i != inputs.size() - 1) {
      oss << ", ";
    }
  }
  oss << ")\n";
  oss << "    # " << task_info->python_stack;
  std::string str = oss.str();
  return str;
}

void GraphTracker::RaceCheck() {
  MS_LOG(WARNING) << "Begin race check";
  auto race_checker = RaceChecker(GetAllAddresses(), GetStreamSize());
  for (const auto &op : operators_) {
    MS_EXCEPTION_IF_NULL(op);
    MS_EXCEPTION_IF_NULL(op->task_info);
    auto stream_id = GetTaskInfoStreamId(op->task_info);
    if (op->task_info->node_name == "RecordEvent") {
      auto iter = op->task_info->attrs.find(kEvent);
      if (iter == op->task_info->attrs.end()) {
        MS_LOG(ERROR) << "Event id is not found, task info: " << op->task_info->time_stamp;
        continue;
      }
      race_checker.RecordEvent(stream_id, iter->second);
    } else if (op->task_info->node_name == "WaitEvent") {
      auto iter = op->task_info->attrs.find(kEvent);
      if (iter == op->task_info->attrs.end()) {
        MS_LOG(ERROR) << "Event id is not found, task info: " << op->task_info->time_stamp;
        continue;
      }
      race_checker.WaitEvent(stream_id, iter->second);
    } else {
      // check read and write
      if (op->task_info->node_name == "Reshape") {
        continue;
      }
      // not need to check side effect, because assign output and input are the same address.
      for (size_t i = 0; i < op->inputs.size(); i++) {
        auto input = op->inputs[i];
        MS_EXCEPTION_IF_NULL(input);
        bool read_error = race_checker.CheckRead(input->start_addr, input->end_addr, stream_id);
        if (read_error) {
          MS_LOG(ERROR) << "Read error, task info: " << op->task_info->time_stamp;
        }
      }
      for (size_t i = 0; i < op->outputs.size(); i++) {
        auto output = op->outputs[i];
        MS_EXCEPTION_IF_NULL(output);
        bool write_error = race_checker.CheckWrite(output->start_addr, output->end_addr, stream_id);
        if (write_error) {
          MS_LOG(ERROR) << "Write error, task info: " << op->task_info->time_stamp;
        }
      }
    }
  }
  MS_LOG(WARNING) << "End race check";
}

void GraphTracker::Dump(const std::string &graph_path) {
  std::lock_guard<std::mutex> lock(mutex_);
  RaceCheck();

  static bool is_simple_tracker = common::IsEnableAllocConfig(common::kAllocSimpleTracker);
  if (is_simple_tracker) {
    MS_LOG(WARNING) << "Simple tracker, skip dump";
    return;
  }
  MS_LOG(WARNING) << "Dump graph to file: " << graph_path;
  std::ofstream graph_file(graph_path);
  // dump operators
  size_t log_threshold = operators_.size() / kLogThreshold;
  size_t i = 0;
  for (const auto &op : operators_) {
    i++;
    MS_EXCEPTION_IF_NULL(op);
    graph_file << op->ToString() << std::endl;
    // print log
    if (i > log_threshold) {
      MS_LOG(WARNING) << "GraphTracker Dump progress: " << (i + 1) * kLogPersentage / operators_.size() << "%";
      log_threshold += operators_.size() / kLogThreshold;
    }
  }
  graph_file.close();
}

bool GraphTracker::NeedDump() {
  std::lock_guard<std::mutex> lock(mutex_);
  return !operators_.empty();
}

TrackerTensorPtr GraphTracker::AddTensor(MemBlockInfoPtr mem_block, DeviceMemPtr device_ptr, TypeId dtype,
                                         const ShapeVector &shape, TensorStorageInfoPtr tensor_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (begin_race_check_) {
    MS_LOG(EXCEPTION) << "AddTensor is not allowed when race check is enabled";
  }
  auto tensor = std::make_shared<TrackerTensor>();
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(mem_block);
  tensor->start_time_stamp = mem_block->start_time_stamp;
  // Strict validation, considering the entire memory block as unsafe, without considering offset and stride for now
  tensor->start_addr = reinterpret_cast<uintptr_t>(mem_block->device_addr);
  tensor->end_addr = tensor->start_addr + mem_block->size - 1;  // -1 for end_addr is exclusive
  tensor->shape = shape;
  tensor->dtype = dtype;
  tensor->tensor_info = tensor_info;
  tensors_.push_back(tensor);
  return tensor;
}

void GraphTracker::AddOperator(TaskInfoPtr task_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (begin_race_check_) {
    MS_LOG(EXCEPTION) << "AddOperator is not allowed when race check is enabled";
  }
  auto op = std::make_shared<TrackerOperator>();
  MS_EXCEPTION_IF_NULL(op);
  op->task_info = task_info;
  op->stream_id = GetTaskInfoStreamId(task_info);
  operators_.push_back(op);
  task_operator_map_[task_info] = op;
}

void GraphTracker::CacheLastTask() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (operators_.empty()) {
    cache_ = nullptr;
    return;
  }
  cache_ = operators_.back();
  operators_.pop_back();
}

void GraphTracker::EmptyCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (cache_ == nullptr) {
    return;
  }
  operators_.push_back(cache_);
  cache_ = nullptr;
}

void GraphTracker::AddOperatorInput(TaskInfoPtr task_info, TrackerTensorPtr tensor) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (begin_race_check_) {
    MS_LOG(EXCEPTION) << "AddOperatorInput is not allowed when race check is enabled";
  }
  auto iter = task_operator_map_.find(task_info);
  if (iter == task_operator_map_.end()) {
    MS_LOG(EXCEPTION) << "Task info " << task_info->time_stamp << " not found";
  }
  auto op = iter->second;
  op->inputs.push_back(tensor);
}

void GraphTracker::AddOperatorOutput(TaskInfoPtr task_info, TrackerTensorPtr tensor) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (begin_race_check_) {
    MS_LOG(EXCEPTION) << "AddOperatorOutput is not allowed when race check is enabled";
  }
  auto iter = task_operator_map_.find(task_info);
  if (iter == task_operator_map_.end()) {
    MS_LOG(EXCEPTION) << "Task info " << task_info->time_stamp << " not found";
  }
  auto op = iter->second;
  op->outputs.push_back(tensor);
}

// for race checker
std::vector<uintptr_t> GraphTracker::GetAllAddresses() {
  begin_race_check_ = true;
  std::vector<uintptr_t> addresses;
  for (const auto &tensor : tensors_) {
    addresses.push_back(tensor->start_addr);
    addresses.push_back(tensor->end_addr);
  }
  return addresses;
}

int32_t GraphTracker::GetStreamSize() {
  begin_race_check_ = true;
  size_t stream_num = 0;
  for (const auto &op : operators_) {
    MS_EXCEPTION_IF_NULL(op);
    MS_EXCEPTION_IF_NULL(op->task_info);
    size_t stream_id = GetTaskInfoStreamId(op->task_info);
    stream_num = std::max(stream_num, stream_id);
  }
  return stream_num + 1;
}
}  // namespace graph
}  // namespace tracker
}  // namespace device
}  // namespace mindspore
