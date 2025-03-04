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

#include "include/backend/mem_reuse/mem_tracker.h"

#include <fstream>
#include <memory>
#include <mutex>
#include <numeric>
#include <string>
#include <unordered_map>

#include "include/backend/mem_reuse/dynamic_mem_pool.h"
#include "include/backend/mem_reuse/mem_pool_util.h"
#include "ir/dtype.h"
#include "utils/log_adapter.h"
#include "utils/ms_context.h"
#include "utils/distributed_meta.h"
#include "include/common/debug/common.h"
#include "include/backend/device_type.h"
#include "include/backend/mem_reuse/mem_dynamic_allocator.h"
#include "include/common/utils/utils.h"
#include "utils/file_utils.h"

namespace mindspore {
namespace device {
namespace tracker {
constexpr int64_t kIllegalStartTimeStamp = -1L;
namespace {

AllocatorType GetAllocatorType(MemType mem_type) {
  static std::map<MemType, device::AllocatorType> mem_allocator_type_map = {
    {MemType::kWeight, AllocatorType::kWeight},
    {MemType::kConstantValue, AllocatorType::kConstantValue},
    {MemType::kKernel, AllocatorType::kConstantValue},
    {MemType::kGraphOutput, AllocatorType::kGraphOutput},
    {MemType::kSomas, AllocatorType::kConstantValue},
    {MemType::kSomasOutput, AllocatorType::kKernelOutput},
    {MemType::kGeConst, AllocatorType::kConstantValue},
    {MemType::kGeFixed, AllocatorType::kOther},
    {MemType::kBatchMemory, AllocatorType::kConstantValue},
    {MemType::kContinuousMemory, AllocatorType::kConstantValue},
    {MemType::kPyNativeInput, AllocatorType::kConstantValue},
    {MemType::kPyNativeOutput, AllocatorType::kKernelOutput},
    {MemType::kWorkSpace, AllocatorType::kWorkspace},
    {MemType::kOther, AllocatorType::kOther}};

  auto iter = mem_allocator_type_map.find(mem_type);
  if (iter == mem_allocator_type_map.end()) {
    MS_LOG(WARNING) << "Not found mem_type:" << mem_type << " in mem_allocator_type_map.";
    return AllocatorType::kOther;
  }
  return iter->second;
}

bool IsPyNative() {
  static bool is_pynative = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE) == kPynativeMode;
  // PythonStack is no need in graph mode.
  return is_pynative;
}

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
}  // namespace

namespace graph {
std::string TrackerTensor::ToString() {
  MS_EXCEPTION_IF_NULL(mem_block);
  return "%" + std::to_string(mem_block->start_time_stamp);
}

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

void TrackerOperator::ValidateMemoryUsage(const std::vector<std::vector<size_t>> &dep) {
  MS_EXCEPTION_IF_NULL(task_info);
  if (name() == "Reshape") {
    return;
  }
  size_t stream_id = GetTaskInfoStreamId(task_info);
  for (size_t i = 0; i < inputs.size(); i++) {
    MS_EXCEPTION_IF_NULL(inputs[i]);
    MS_EXCEPTION_IF_NULL(inputs[i]->mem_block);
    if (inputs[i]->mem_block->end_time_stamp <= task_info->time_stamp) {
      MS_LOG(WARNING) << "Valid failed: Input tensor " << inputs[i]->ToString() << " is not valid for operator "
                      << name() << ", task info: " << task_info->time_stamp;
    }
    auto last_write_time_stamp = inputs[i]->mem_block->last_write_time_stamp;
    auto last_write_stream_id = inputs[i]->mem_block->last_write_stream_id;
    if (last_write_stream_id != stream_id && last_write_time_stamp > dep[stream_id][last_write_stream_id]) {
      MS_LOG(WARNING) << "Valid failed: Input tensor " << inputs[i]->ToString() << " is not valid for operator "
                      << name() << ", task info: " << task_info->time_stamp
                      << ", maybe the input tensor is not ready by event.";
    }
  }
  for (size_t i = 0; i < outputs.size(); i++) {
    MS_EXCEPTION_IF_NULL(outputs[i]);
    MS_EXCEPTION_IF_NULL(outputs[i]->mem_block);
    if (outputs[i]->mem_block->end_time_stamp <= task_info->time_stamp) {
      MS_LOG(WARNING) << "Valid failed: Output tensor " << outputs[i]->ToString() << " is not valid for operator "
                      << name() << ", task info: " << task_info->time_stamp;
    }
    outputs[i]->mem_block->last_write_time_stamp = static_cast<size_t>(task_info->time_stamp);
    outputs[i]->mem_block->last_write_stream_id = stream_id;
  }
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

void MultiStreamDependency::Init(size_t stream_num) {
  dependency.clear();
  dependency.resize(stream_num);
  for (size_t i = 0; i < stream_num; i++) {
    dependency[i].resize(stream_num, 0);
  }
}

void MultiStreamDependency::RecordEvent(size_t stream_id, const std::string &event_id, size_t time_stamp) {
  if (stream_id >= dependency.size()) {
    MS_LOG(ERROR) << "Stream id " << stream_id << " is out of range.";
    return;
  }
  dependency[stream_id][stream_id] = time_stamp;
  event_map[event_id] = dependency[stream_id];
}

void MultiStreamDependency::WaitEvent(size_t stream_id, const std::string &event_id) {
  auto iter = event_map.find(event_id);
  if (iter == event_map.end()) {
    MS_LOG(ERROR) << "Event id " << event_id << " is not found.";
    return;
  }
  if (stream_id >= dependency.size()) {
    MS_LOG(ERROR) << "Stream id " << stream_id << " is out of range.";
    return;
  }
  for (size_t i = 0; i < dependency[stream_id].size(); i++) {
    dependency[stream_id][i] = std::max(dependency[stream_id][i], iter->second[i]);
  }
}

void GraphTracker::Dump(const std::string &graph_path) {
  MS_LOG(WARNING) << "Dump graph to file: " << graph_path;
  ChangeFileMode(graph_path, S_IWUSR | S_IRUSR);
  std::ofstream graph_file(graph_path);
  InitStreamSize();
  // dump operators
  for (const auto &op : operators_) {
    MS_EXCEPTION_IF_NULL(op);
    op->ValidateMemoryUsage(dep_.dependency);
    graph_file << op->ToString() << std::endl;
    MS_EXCEPTION_IF_NULL(op->task_info);
    auto stream_id = GetTaskInfoStreamId(op->task_info);
    if (op->task_info->node_name == "RecordEvent") {
      auto iter = op->task_info->attrs.find(kEvent);
      if (iter == op->task_info->attrs.end()) {
        MS_LOG(ERROR) << "Event id is not found.";
        continue;
      }
      dep_.RecordEvent(stream_id, iter->second, op->task_info->time_stamp);
    } else if (op->task_info->node_name == "WaitEvent") {
      auto iter = op->task_info->attrs.find(kEvent);
      if (iter == op->task_info->attrs.end()) {
        MS_LOG(ERROR) << "Event id is not found.";
        continue;
      }
      dep_.WaitEvent(stream_id, iter->second);
    }
  }
  graph_file.close();
}

void GraphTracker::InitStreamSize() {
  size_t stream_num = 0;
  for (const auto &op : operators_) {
    MS_EXCEPTION_IF_NULL(op);
    MS_EXCEPTION_IF_NULL(op->task_info);
    size_t stream_id = GetTaskInfoStreamId(op->task_info);
    stream_num = std::max(stream_num, stream_id);
  }
  dep_.Init(stream_num + 1);
}

TrackerTensorPtr GraphTracker::AddTensor(MemBlockInfoPtr mem_block, TypeId dtype, const ShapeVector &shape,
                                         TensorStorageInfoPtr tensor_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto tensor = std::make_shared<TrackerTensor>();
  MS_EXCEPTION_IF_NULL(tensor);
  MS_EXCEPTION_IF_NULL(mem_block);
  tensor->mem_block = mem_block;
  tensors_.push_back(tensor);
  tensor->shape = shape;
  tensor->dtype = dtype;
  tensor->tensor_info = tensor_info;
  return tensor;
}

void GraphTracker::AddOperator(TaskInfoPtr task_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto op = std::make_shared<TrackerOperator>();
  MS_EXCEPTION_IF_NULL(op);
  op->task_info = task_info;
  operators_.push_back(op);
  task_operator_map_[task_info] = op;
}

void GraphTracker::CacheLastTask() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (operators_.empty()) {
    cache = nullptr;
    return;
  }
  cache = operators_.back();
  operators_.pop_back();
}

void GraphTracker::EmptyCache() {
  std::lock_guard<std::mutex> lock(mutex_);
  if (cache == nullptr) {
    return;
  }
  operators_.push_back(cache);
  cache = nullptr;
}

TrackerOperatorPtr GraphTracker::GetOperator(TaskInfoPtr task_info) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = task_operator_map_.find(task_info);
  if (iter == task_operator_map_.end()) {
    return nullptr;
  }
  return iter->second;
}

}  // namespace graph

std::tuple<std::string, std::string, std::string> MemoryTrackerEnabled::GetPath(size_t rank_id) {
  std::string block_csv_path = memory::mem_pool::GeneratePath(rank_id, "/memory_block", "csv");
  std::string task_csv_path = memory::mem_pool::GeneratePath(rank_id, "/task", "csv");
  std::string graph_path = memory::mem_pool::GeneratePath(rank_id, "/tracker_graph", "ir");
  return std::tuple(block_csv_path, task_csv_path, graph_path);
}

void MemoryTrackerEnabled::AddTask(const std::string &task_name, const std::string &node_name,
                                   const std::string &graph_name, const bool to_graph, const std::string &file_name,
                                   size_t line_num) {
  std::string python_stack;
  if (IsPyNative()) {
    python_stack = GetPythonStackStr();
  }

  std::lock_guard lock(mutex_);
  if (!is_init_enable_hccl_) {
    // MS_CTX_ENABLE_HCCL will be reset when the process is destroyed.
    // Therefore, record the enable_hccl when AddTask for the first time.
    auto ms_context = MsContext::GetInstance();
    enable_hccl_ = ms_context->get_param<bool>(MS_CTX_ENABLE_HCCL);
    is_init_enable_hccl_ = true;
  }

  auto task_info = std::make_shared<TaskInfo>();
  MS_EXCEPTION_IF_NULL(task_info);
  task_info->task_name = task_name;
  task_info->node_name = node_name;
  task_info->graph_name = graph_name;
  task_info->file_name = file_name;
  task_info->line_num = line_num;
  task_info->time_stamp = time_stamp_;
  task_info->python_stack = python_stack;
  task_info->attrs[kStreamId] = "0";
  task_map_[task_name] = task_info;
  if (to_graph) {
    time_stamp_++;
    task_list_.push_back(task_info);
    graph::GraphTracker::getInstance().AddOperator(task_info);
  }
}
void MemoryTrackerEnabled::AddTask(const std::string &task_name, const std::string &node_name,
                                   const std::string &graph_name, const std::string &file_name, size_t line_num) {
  AddTask(task_name, node_name, graph_name, true, file_name, line_num);
}

void MemoryTrackerEnabled::UpdateTask(const std::string &task_name,
                                      const std::unordered_map<std::string, std::string> &attrs) {
  std::lock_guard lock(mutex_);
  auto iter = task_map_.find(task_name);
  if (iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker UpdateTask failed, task_name:" << task_name << " not found";
    return;
  }
  for (const auto &attr : attrs) {
    iter->second->attrs[attr.first] = attr.second;
  }
}

void MemoryTrackerEnabled::CacheLastTask() {
  std::lock_guard lock(mutex_);
  if (task_list_.empty()) {
    cache = nullptr;
    return;
  }
  cache = task_list_.back();
  task_list_.pop_back();
  graph::GraphTracker::getInstance().CacheLastTask();
}

void MemoryTrackerEnabled::EmptyCache() {
  std::lock_guard lock(mutex_);
  if (cache == nullptr) {
    return;
  }
  task_list_.push_back(cache);
  cache = nullptr;
  graph::GraphTracker::getInstance().EmptyCache();
}

void MemoryTrackerEnabled::AddNestedTask(const std::string &task_name, const std::string &node_name,
                                         const std::string &graph_name, const std::string &file_name, size_t line_num) {
  {
    // lock scope
    std::lock_guard lock(mutex_);
    nested_num_++;
    if (nested_num_ != 1) {
      return;
    }
  }
  AddTask(task_name, node_name, graph_name, file_name, line_num);
}

void MemoryTrackerEnabled::DelNestedTask() {
  std::lock_guard lock(mutex_);
  nested_num_--;
}

void MemoryTrackerEnabled::MarkTensorAsInput(const std::string &task_name, const std::string &device_name,
                                             DeviceMemPtr device_ptr, TypeId dtype, const ShapeVector &shape,
                                             TensorStorageInfoPtr tensor_info, const std::string &file_name,
                                             size_t line_num) {
  if (device_name == "CPU" || device_ptr == nullptr) {
    return;
  }
  UseMemBlock(task_name, device_ptr, file_name, line_num);
  std::lock_guard lock(mutex_);
  if (nested_num_ > 1) {
    return;
  }
  auto iter = task_map_.find(task_name);
  if (iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker MarkTensorAsInput failed, task_name:" << task_name << " not found";
    return;
  }
  auto task_info = iter->second;
  auto mem_block_iter = FindMemBlock(device_ptr, file_name, line_num);
  if (mem_block_iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker MarkTensorAsInput failed, device_ptr:" << device_ptr << " not found";
    return;
  }
  auto mem_block = mem_block_iter->second;
  auto input_tensor = graph::GraphTracker::getInstance().AddTensor(mem_block, dtype, shape, tensor_info);
  auto op = graph::GraphTracker::getInstance().GetOperator(task_info);
  if (op == nullptr) {
    MS_LOG(ERROR) << "MemoryTracker MarkTensorAsInput failed, task_name:" << task_name << " not found";
    return;
  }
  op->inputs.push_back(input_tensor);
}

void MemoryTrackerEnabled::MarkTensorAsOutput(const std::string &task_name, const std::string &device_name,
                                              DeviceMemPtr device_ptr, TypeId dtype, const ShapeVector &shape,
                                              TensorStorageInfoPtr tensor_info, const std::string &file_name,
                                              size_t line_num) {
  std::lock_guard lock(mutex_);
  if (device_name == "CPU" || device_ptr == nullptr) {
    return;
  }
  if (nested_num_ > 1) {
    return;
  }
  auto iter = task_map_.find(task_name);
  if (iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker MarkTensorAsOutput failed, task_name:" << task_name << " not found";
    return;
  }
  auto task_info = iter->second;
  auto mem_block_iter = FindMemBlock(device_ptr, file_name, line_num);
  if (mem_block_iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker MarkTensorAsOutput failed, device_ptr:" << device_ptr << " not found";
    return;
  }
  auto mem_block = mem_block_iter->second;
  auto output_tensor = graph::GraphTracker::getInstance().AddTensor(mem_block, dtype, shape, tensor_info);
  auto op = graph::GraphTracker::getInstance().GetOperator(task_info);
  if (op == nullptr) {
    MS_LOG(ERROR) << "MemoryTracker MarkTensorAsOutput failed, task_name:" << task_name << " not found";
    return;
  }
  op->outputs.push_back(output_tensor);
}

MemInfoPtr MemoryTrackerEnabled::NewMemInfo(const std::string &task_name, MemType type, size_t size,
                                            KernelTensorPtr kernel_tensor, const std::string &file_name,
                                            size_t line_num) {
  auto mem_info = std::make_shared<MemInfo>();
  MS_EXCEPTION_IF_NULL(mem_info);
  mem_info->type = type;
  mem_info->size = size;
  mem_info->kernel_tensor = kernel_tensor;
  mem_info->file_name = file_name;
  mem_info->line_num = line_num;
  auto iter = task_map_.find(task_name);
  if (iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker AddMemInfo failed, task_name:" << task_name << " not found, " << file_name << ":"
                  << line_num;
    return nullptr;
  }

  const auto &node_name = iter->second->node_name;
  DynamicMemAllocatorDebugInfo::SetDebugInfo(node_name, GetAllocatorType(type));

  mem_info->producer_task = iter->second;
  mem_info_list_.push_back(mem_info);
  return mem_info;
}

void MemoryTrackerEnabled::AddMemInfoForKernelTensor(const std::string &task_name, MemType type, size_t size,
                                                     KernelTensorPtr kernel_tensor, const std::string &file_name,
                                                     size_t line_num) {
  auto mem_info = NewMemInfo(task_name, type, size, kernel_tensor, file_name, line_num);
  if (mem_info != nullptr) {
    kernel_tensor_mem_map[kernel_tensor] = mem_info;
  }
}

void MemoryTrackerEnabled::AddMemInfo(const std::string &task_name, MemType type, size_t size,
                                      DeviceAddress *device_address, const std::string &file_name, size_t line_num) {
  MS_EXCEPTION_IF_NULL(device_address);
  if (device_address->GetDeviceType() == DeviceType::kCPU) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);

  if (device_address->kernel_tensor() == nullptr) {
    auto mem_info = NewMemInfo(task_name, type, size, nullptr, file_name, line_num);
    device_address_mem_map[device_address] = mem_info;
  } else {
    AddMemInfoForKernelTensor(task_name, type, size, device_address->kernel_tensor().get(), file_name, line_num);
  }
}

void MemoryTrackerEnabled::AddCompileTimeMemInfo(const std::string &task_name, size_t size, DeviceMemPtr device_ptr,
                                                 MemType mem_type, const std::string &file_name, size_t line_num) {
  std::lock_guard lock(mutex_);
  auto mem_info = std::make_shared<MemInfo>();
  MS_EXCEPTION_IF_NULL(mem_info);
  mem_info->type = mem_type;
  mem_info->size = size;
  mem_info->file_name = file_name;
  mem_info->line_num = line_num;
  auto iter = task_map_.find(task_name);
  if (iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker AddCompileTimeMemInfo failed, task_name:" << task_name << " not found, "
                  << file_name << ":" << line_num;
    return;
  }
  mem_info->producer_task = iter->second;
  auto mem_block_iter = FindMemBlock(device_ptr, file_name, line_num);
  if (mem_block_iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker AddCompileTimeMemInfo failed, device_ptr:" << device_ptr << " not found, "
                  << file_name << ":" << line_num;
    return;
  }
  mem_info->mem_block = mem_block_iter->second;
  mem_info->mem_block->is_bind = true;
  mem_info->mem_block->mem_info = mem_info;
  mem_info_list_.push_back(mem_info);
}

void MemoryTrackerEnabled::BindDevicePtr(DeviceAddress *device_address, DeviceMemPtr device_ptr,
                                         const std::string &file_name, size_t line_num) {
  if (device_address == nullptr) {
    return;
  }
  std::lock_guard<std::mutex> lock(mutex_);
  if (device_address->GetDeviceType() == DeviceType::kCPU) {
    return;
  }
  MemInfoPtr mem_info{nullptr};
  if (device_address->kernel_tensor() == nullptr) {
    auto iter = device_address_mem_map.find(device_address);
    if (iter == device_address_mem_map.end()) {
      MS_LOG(ERROR) << "MemoryTracker BindDevicePtr failed, device_address:" << device_address << " not found, "
                    << file_name << ":" << line_num;
      return;
    }
    mem_info = iter->second;
  } else {
    auto iter = kernel_tensor_mem_map.find(device_address->kernel_tensor().get());
    if (iter == kernel_tensor_mem_map.end()) {
      MS_LOG(WARNING) << "MemoryTracker BindDevicePtr failed, device_address : " << device_address
                      << ", kernel_tensor:" << device_address->kernel_tensor().get() << " not found, " << file_name
                      << ":" << line_num;
      return;
    }
    mem_info = iter->second;
  }
  auto mem_block_iter = FindMemBlock(device_ptr, file_name, line_num);
  if (mem_block_iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker BindDevicePtr failed, device_addr:" << device_ptr << " not found, " << file_name
                  << ":" << line_num;
    return;
  }
  mem_info->mem_block = mem_block_iter->second;
  mem_info->mem_block->is_bind = true;
  mem_info->mem_block->mem_info = mem_info;
}

void MemoryTrackerEnabled::AllocMemBlock(DeviceMemPtr device_addr, size_t size, const std::string &pool_name,
                                         size_t actual_peak_memory, size_t in_used_size, size_t total_size,
                                         uint32_t stream_id) {
  std::lock_guard lock(mutex_);
  time_stamp_++;
  auto mem_block = std::make_shared<MemBlockInfo>();
  MS_EXCEPTION_IF_NULL(mem_block);
  mem_block->device_addr = device_addr;
  mem_block->start_time_stamp = time_stamp_;
  mem_block->actual_peak_memory = actual_peak_memory;
  mem_block->size = size;
  mem_block->pool_name = pool_name;
  mem_block->stream_id = stream_id;
  mem_block->real_start_time = GetCurrentUSec();
  mem_block->alloc_in_used_size = in_used_size;
  mem_block->alloc_total_size = total_size;
  device_mem_block_map[device_addr] = mem_block;
  mem_block_list_.emplace_back(mem_block);
  // mem_block need to dump again, after mem_block_list_ changed
  has_dump = false;
}

std::map<DeviceMemPtr, MemBlockInfoPtr>::iterator MemoryTrackerEnabled::FindMemBlock(DeviceMemPtr device_ptr,
                                                                                     const std::string &file_name,
                                                                                     size_t line_num) {
  auto it = device_mem_block_map.upper_bound(device_ptr);
  if (it == device_mem_block_map.begin()) {
    return device_mem_block_map.end();
  }
  --it;
  DeviceMemPtr right_border = static_cast<const uint8_t *>(it->first) + it->second->size;
  if (device_ptr < right_border) {
    return it;
  }
  MS_LOG(ERROR) << "MemoryTracker FindMemBlock failed, device_ptr:" << device_ptr
                << " not found, right border: " << right_border << ", " << file_name << ":" << line_num;
  return device_mem_block_map.end();
}

void MemoryTrackerEnabled::FreeMemBlock(DeviceMemPtr device_addr, size_t in_used_size, size_t total_size) {
  std::lock_guard lock(mutex_);
  time_stamp_++;
  auto iter = device_mem_block_map.find(device_addr);
  if (iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker FreeMemBlock failed, device_addr:" << device_addr << " not found";
    return;
  }
  iter->second->end_time_stamp = time_stamp_;
  iter->second->real_end_time = GetCurrentUSec();
  iter->second->release_in_used_size = in_used_size;
  iter->second->release_total_size = total_size;
  device_mem_block_map.erase(iter);
}

void MemoryTrackerEnabled::UseMemBlock(const std::string &task_name, DeviceMemPtr device_addr,
                                       const std::string &file_name, size_t line_num) {
  std::lock_guard lock(mutex_);
  auto iter = FindMemBlock(device_addr, file_name, line_num);
  if (iter == device_mem_block_map.end()) {
    MS_LOG(ERROR) << "MemoryTracker UseMemBlock failed, device_addr:" << device_addr << " not found, " << file_name
                  << ":" << line_num;
    return;
  }
  if (iter->second->pool_name == "CPU") {
    return;
  }
  auto task_iter = task_map_.find(task_name);
  if (task_iter == task_map_.end()) {
    MS_LOG(ERROR) << "MemoryTracker UseMemBlock failed, task_name:" << task_name << " not found, " << file_name << ":"
                  << line_num;
    return;
  }
  auto mem_info = iter->second->mem_info.lock();
  if (mem_info == nullptr) {
    MS_LOG(ERROR) << "MemoryTracker UseMemBlock failed, mem_info is null, " << file_name << ":" << line_num
                  << ", addr:" << device_addr;
    return;
  }
  mem_info->user_tasks.push_back(task_iter->second);
}

namespace {
constexpr size_t kKBToByte = 1024;
constexpr size_t kMBToKB = 1024;
static const int kPrecisionDigits = 20;

auto task_list_to_str = [](const std::vector<TaskInfoPtr> &task_list) -> std::string {
  std::stringstream ss;
  ss << "{";
  std::string result = "";
  if (!task_list.empty()) {
    result = std::accumulate(
      std::next(task_list.begin()), task_list.end(), std::to_string(task_list.front()->time_stamp),
      [](const std::string &ss, TaskInfoPtr task) { return ss + "-" + std::to_string(task->time_stamp); });
  }
  ss << result;
  ss << "}";
  return ss.str();
};

const std::vector<std::pair<std::string, std::function<void(const MemBlockInfoPtr &, std::ofstream &)>>> block_csv = {
  {"start_time_stamp",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->start_time_stamp; }},
  {"end_time_stamp", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->end_time_stamp; }},
  {"device_addr", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->device_addr; }},
  {"stream_id", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->stream_id; }},
  {"pool_type", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->pool_name; }},
  {"size", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->size; }},
  {"actual_peak_memory",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->actual_peak_memory; }},
  {"file_name",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       oss << mem_info->file_name;
     }
   }},
  {"line_num",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       oss << mem_info->line_num;
     }
   }},
  {"type",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       oss << MemTypeToStr.at(mem_info->type);
     }
   }},
  {"producer_task",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->time_stamp;
     }
   }},
  {"task_name",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->task_name;
     }
   }},
  {"node_name",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->node_name;
     }
   }},
  {"graph_name",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->graph_name;
     }
   }},
  {"user_tasks",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       oss << task_list_to_str(mem_info->user_tasks);
     }
   }},
  {"last_user_task",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       std::string result = "";
       if (!mem_info->user_tasks.empty()) {
         result = std::to_string(mem_info->user_tasks.back()->time_stamp);
       }
       oss << result;
     }
   }},
  {"python_stack",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->python_stack;
     }
   }},
};

const std::vector<std::pair<std::string, std::function<void(const TaskInfoPtr &, std::ofstream &)>>> task_csv = {
  {"time_stamp", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->time_stamp; }},
  {"task_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->task_name; }},
  {"node_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->node_name; }},
  {"graph_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->graph_name; }},
  {"file_name", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->file_name; }},
  {"line_num", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->line_num; }},
  {"python_stack", [](const TaskInfoPtr &task, std::ofstream &oss) { oss << task->python_stack; }},
};

const std::vector<std::pair<std::string, std::function<void(const MemBlockInfoPtr &, std::ofstream &)>>> prof_csv = {
  {"Name",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     auto mem_info = mem_block->mem_info.lock();
     if (mem_info) {
       MS_EXCEPTION_IF_NULL(mem_info->producer_task);
       oss << mem_info->producer_task->node_name;
     }
   }},
  {"Size(KB)", [](const MemBlockInfoPtr &mem_block,
                  std::ofstream &oss) { oss << (static_cast<float>(mem_block->size) / kKBToByte); }},
  {"Allocation Time(us)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->real_start_time; }},
  {"Duration(us)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     if (mem_block->real_end_time > 0) {
       oss << (mem_block->real_end_time - mem_block->real_start_time);
     }
   }},
  {"Allocation Total Allocated(MB)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     oss << (static_cast<float>(mem_block->alloc_in_used_size) / kKBToByte / kMBToKB);
   }},
  {"Allocation Total Reserved(MB)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     oss << (static_cast<float>(mem_block->alloc_total_size) / kKBToByte / kMBToKB);
   }},
  {"Release Total Allocated(MB)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     oss << (static_cast<float>(mem_block->release_in_used_size) / kKBToByte / kMBToKB);
   }},
  {"Release Total Reserved(MB)",
   [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) {
     oss << (static_cast<float>(mem_block->release_total_size) / kKBToByte / kMBToKB);
   }},
  {"Device", [](const MemBlockInfoPtr &mem_block, std::ofstream &oss) { oss << mem_block->pool_name; }},
};
}  // namespace

void MemoryTrackerEnabled::Dump(size_t rank_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (has_dump) {
    return;
  }
  has_dump = true;

  auto [block_csv_path, task_csv_path, graph_path] = GetPath(rank_id);
  if (block_csv_path.empty() || task_csv_path.empty() || graph_path.empty()) {
    MS_LOG(ERROR) << "Get realpath failed, block_csv_path:" << block_csv_path << ", task_csv_path:" << task_csv_path
                  << ", " << graph_path;
    return;
  }

  graph::GraphTracker::getInstance().Dump(graph_path);

  MS_LOG(INFO) << "MemoryTracker Dump start";
  MS_LOG(WARNING) << "block csv path: " << block_csv_path;
  MS_LOG(WARNING) << "task csv path: " << task_csv_path;
  std::ofstream block_file(block_csv_path);
  if (!block_file) {
    MS_LOG(EXCEPTION) << "Open file " << block_csv_path << " failed.";
  }
  size_t not_bind_size = 0;
  for (const auto &csv : block_csv) {
    block_file << csv.first << ",";
  }
  block_file << "\n";
  for (auto &mem_block : mem_block_list_) {
    if (mem_block->pool_name == "CPU") {
      continue;
    }
    for (const auto &csv : block_csv) {
      csv.second(mem_block, block_file);
      block_file << ",";
    }
    if (!mem_block->is_bind) {
      not_bind_size += mem_block->size;
    }
    block_file << "\n";
  }

  std::ofstream task_file(task_csv_path);
  if (!task_file) {
    MS_LOG(EXCEPTION) << "Open file " << task_csv_path << " failed.";
  }
  for (const auto &csv : task_csv) {
    task_file << csv.first << ",";
  }
  task_file << "\n";
  for (auto &task : task_list_) {
    for (const auto &csv : task_csv) {
      csv.second(task, task_file);
      task_file << ",";
    }
    task_file << "\n";
  }

  block_file.close();
  task_file.close();
  MS_LOG(INFO) << "Not bind size, " << not_bind_size;
  MS_LOG(INFO) << "MemoryTracker Dump end";
}

void MemoryTrackerEnabled::UpdateProfilingPos() {
  std::lock_guard<std::mutex> lock(mutex_);
  last_profiling_pos_ = mem_block_list_.size();
}

void MemoryTrackerEnabled::DumpProfilingMemInfo(size_t rank_id, const std::string &path, const std::string &file_name) {
  std::lock_guard<std::mutex> lock(mutex_);

  auto csv_path = path + "/" + file_name + "_" + std::to_string(rank_id) + ".csv";
  auto csv_path_opt = Common::CreatePrefixPath(csv_path);
  if (!csv_path_opt.has_value()) {
    MS_LOG(ERROR) << "Get realpath failed, csv_path:" << csv_path;
    return;
  }

  MS_LOG(INFO) << "MemoryTracker DumpProfilingMemInfo start, last_profiling_pos:" << last_profiling_pos_;
  ChangeFileMode(csv_path_opt.value(), S_IWUSR | S_IRUSR);
  std::ofstream block_file(csv_path_opt.value());
  auto old_file_flags = block_file.flags();
  auto old_precision = block_file.precision();
  block_file.unsetf(std::ios_base::floatfield);
  block_file.precision(kPrecisionDigits);
  for (const auto &csv : prof_csv) {
    block_file << csv.first << ",";
  }
  block_file << "\n";

  for (size_t i = 0; i < mem_block_list_.size(); i++) {
    const auto &mem_block = mem_block_list_[i];
    if (i < last_profiling_pos_) {
      continue;
    }

    if (mem_block->pool_name == "CPU") {
      continue;
    }

    if (mem_block->start_time_stamp == kIllegalStartTimeStamp) {
      MS_LOG(DEBUG) << "Mem block start time stamp is " << kIllegalStartTimeStamp << ".";
      continue;
    }

    for (const auto &csv : prof_csv) {
      csv.second(mem_block, block_file);
      block_file << ",";
    }
    block_file << "\n";
  }

  // Restore file flags and precision
  block_file.flags(old_file_flags);
  block_file.precision(old_precision);
  block_file.close();
  ChangeFileMode(csv_path_opt.value(), S_IWUSR | S_IRUSR);

  // record the last time stamp
  last_profiling_pos_ = mem_block_list_.size();
  MS_LOG(INFO) << "MemoryTracker DumpProfilingMemInfo end, last_profiling_pos:" << last_profiling_pos_;
}

}  // namespace tracker
}  // namespace device
}  // namespace mindspore
