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
#include "runtime/graph_scheduler/graph_parameter_store.h"

#include <algorithm>
#include <string>
#include "runtime/graph_scheduler/device_tensor_copy_store.h"
#include "runtime/graph_scheduler/actor/actor_common.h"
#include "runtime/graph_scheduler/graph_capture/graph_capture_manager.h"
#include "runtime/hardware/device_context_manager.h"
#include "utils/ms_context.h"
#include "utils/llm_manager.h"
#include "runtime/device/res_manager/hal_res_manager.h"

namespace mindspore {
namespace runtime {
void GraphParameterStore::SetPositionWeight(size_t outer_index, bool is_weight) {
  if (outer_index >= is_weights_.size()) {
    MS_LOG(EXCEPTION) << "Outer index is larger than the size of is weights [" << is_weights_.size() << "].";
  }

  if (!is_weights_[outer_index] && is_weight) {
    weight_num_++;
  }
  is_weights_[outer_index] = is_weight;
}

bool GraphParameterStore::GetPositionWeight(size_t outer_index) {
  if (outer_index >= is_weights_.size()) {
    MS_LOG(ERROR) << "Index " << outer_index << ", is out of range of outer size: " << is_weights_.size();
    return false;
  }
  return is_weights_[outer_index];
}

void GraphParameterStore::SetDeviceTensorPrepared(size_t outer_idx, size_t inner_idx, bool is_prepared) {
  auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_idx][inner_idx];
  kernel_tensor_with_info.second.second = is_prepared;
}

bool GraphParameterStore::GetDeviceTensorPrepared(size_t outer_idx, size_t inner_idx) {
  auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_idx][inner_idx];
  return kernel_tensor_with_info.second.second;
}

size_t GraphParameterStore::GetNonWeightParameterNum() { return is_weights_.size() - weight_num_; }

void GraphParameterStore::SetPositionTensor(size_t outer_index, bool is_tensor) {
  if (outer_index >= is_tensors_.size()) {
    MS_LOG(ERROR) << "Index " << outer_index << ", is out of range of outer size: " << is_tensors_.size();
    return;
  }
  is_tensors_[outer_index] = is_tensor;
}

bool GraphParameterStore::GetPositionTensor(size_t outer_index) { return is_tensors_[outer_index]; }

void GraphParameterStore::SetParameterUsedTimes(size_t outer_index, size_t inner_index, size_t times) {
  parameter_used_times_[outer_index][inner_index] = times;
}

bool GraphParameterStore::IsConcurrentlyUse(size_t outer_index, size_t inner_index) {
  return parameter_used_times_[outer_index][inner_index] > 1;
}

void GraphParameterStore::SetFrontNodeToIndex(AnfNode *node, size_t index) {
  MS_EXCEPTION_IF_NULL(node);
  const auto &iter = front_node_to_index_.find(node);
  if (iter != front_node_to_index_.end()) {
    MS_LOG(INFO) << "Update index for front node " << node->DebugString() << " in graph parameter store.";
    iter->second = index;
  }
  front_node_to_index_.emplace(node, index);
  index_to_front_node_.emplace(index, node);
}

void GraphParameterStore::InsertTensorDataIntoCallback(const TensorDataPtr &tensor_data) {
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  tensor_data_in_callback_.push_back(tensor_data);
}

void GraphParameterStore::InsertNonWeightRefMaxInputs(size_t outer_index, size_t inner_index) {
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  non_weight_ref_max_inputs_.emplace(outer_index, inner_index);
}

void GraphParameterStore::ResetPrepareState() {
  for (size_t i = 0; i < parameter_kernel_tensors_.size(); ++i) {
    auto &kernel_tensors = parameter_kernel_tensors_[i];
    for (size_t j = 0; j < kernel_tensors.size(); ++j) {
      kernel_tensors[j].second.second = false;
    }
  }
  tensor_data_in_callback_.reserve(buffer_size_);
}

void GraphParameterStore::ResetAddrRefCount(size_t outer_index, size_t inner_index) {
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
  bool is_ref_count_max = kernel_tensor_with_info.second.first == SIZE_MAX;

  if (kernel_tensor_with_info.first != nullptr) {
    auto &device_tensor = kernel_tensor_with_info.first->device_address();
    if (device_tensor != nullptr) {
      auto user_cnt = kernel_tensor_with_info.second.first;
      if (user_cnt > 0) {
        // When allocate memory, the ref count would be increase, so it should be decrease here.
        if (is_ref_count_max) {
          device_tensor->set_new_ref_count(SIZE_MAX);
        } else {
          static std::string name = "Parameter store";
          device_tensor->IncreaseNewRefCount(name, user_cnt - 1);
        }
        device_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
        MS_LOG(DEBUG) << "Parameter store set new ref count:" << (user_cnt - 1)
                      << " for kernel tensor:" << kernel_tensor_with_info.first->ToString();
      } else {
        MS_LOG(DEBUG) << "User count:0 for parameter store outer index:" << outer_index
                      << " inner index:" << inner_index << " for device address:" << device_tensor;
      }
      return;
    }
  }
}

KernelTensorPtr GraphParameterStore::Fetch(size_t outer_index, size_t inner_index) {
  std::shared_lock<std::shared_mutex> lock(param_mutex_);
  const auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
  return kernel_tensor_with_info.first;
}

const std::function<void(size_t)> &GraphParameterStore::GetAsyncMemcpyFun(size_t outer_index,
                                                                          size_t inner_index) const {
  return async_copy_funcs_[outer_index][inner_index];
}

void GraphParameterStore::SetAsyncMemcpyFun(size_t outer_index, size_t inner_index,
                                            std::function<void(size_t)> &&func) {
  async_copy_funcs_[outer_index][inner_index] = std::move(func);
}

void GraphParameterStore::Push(size_t outer_index, size_t inner_index, const KernelTensorPtr &value, size_t cnt) {
  CheckIndexValid(outer_index, inner_index);
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  MS_LOG(DEBUG) << "Push parameter store for outer index:" << outer_index << " inner index:" << inner_index
                << " count:" << cnt << " kernel tensor:" << value;
  auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
  kernel_tensor_with_info.first = value;
  kernel_tensor_with_info.second.first = cnt;
  if (value->device_address()) {
    parameter_device_names_[outer_index][inner_index] = value->device_address()->device_name();
  }
}

std::string GraphParameterStore::GetParameterDeviceName(size_t outer_index, size_t inner_index) const {
  return parameter_device_names_[outer_index][inner_index];
}

bool GraphParameterStore::CheckBufferSize(size_t outer_index) const {
  std::shared_lock<std::shared_mutex> lock(param_mutex_);
  return buffers_[outer_index].size() > 0;
}

Tensor *GraphParameterStore::FetchTensor(size_t args_index, const KernelWithIndex &node) {
  // Process tensor types that are frequently used to speed up fetchtensor performance.
  if (is_tensors_[args_index]) {
    auto tensor = utils::cast<tensor::TensorPtr>((*input_args_)[args_index]);
    MS_EXCEPTION_IF_NULL(tensor);
    return tensor.get();
  }

  if (args_index >= buffers_.size()) {
    MS_LOG(EXCEPTION) << "Index " << args_index << " is out of buffers range " << buffers_.size() << ".";
  }
  Tensor *tensor = nullptr;
  if (CheckBufferSize(args_index)) {
    std::shared_lock<std::shared_mutex> lock(param_mutex_);
    if (node.second >= buffers_[args_index].size()) {
      MS_LOG(EXCEPTION) << "Node position " << node.second << " is out of buffers position "
                        << buffers_[args_index].size() << " range.";
    }
    tensor = buffers_[args_index][node.second].get();
  } else {
    tensor = FlattenInputTensorByArg(args_index, node);
  }
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor;
}

bool GraphParameterStore::RecordGraphInputsAndIsDyn(const std::vector<size_t> &input_index,
                                                    const std::vector<ParameterPtr> &parameters) {
  bool isDyn = false;
  auto &llm_manager = LLMManager::GetInstance();
  auto enable_capture_graph = GraphCaptureManager::GetInstance().GetEnableGraphCapture();
  for (size_t l = 0; l < input_index.size(); ++l) {
    auto i = input_index[l];
    auto origin_parameter = parameters[l];
    MS_EXCEPTION_IF_NULL(origin_parameter);
    auto buffer_inner_size = buffers_[i].size();
    // List tensor input do not compare shape.
    if (!enable_capture_graph) {
      if (buffer_inner_size != 1) {
        continue;
      }
    }
    const auto &input_tensor = buffers_[i][0];
    if (input_tensor == nullptr) {
      MS_LOG(ERROR) << "The input tensor is nullptr for arg outer index: " << i;
      continue;
    }
    if (!isDyn) {
      if (host_tensors_shape_[i] != input_tensor->shape() || input_tensor->shape().empty()) {
        isDyn = true;
      }
    }
    host_tensors_shape_[i] = input_tensor->shape();
    input_tensor->set_name(origin_parameter->fullname_with_scope());
    llm_manager.add_graph_input(origin_parameter->fullname_with_scope(), input_tensor->data_ptr());
  }
  return isDyn;
}

void AddCopyDataCallBack(const std::vector<TensorDataPtr> &tensor_data_in_callback) {
  device::CallbackFunc callback_func = [tensor_data_in_callback]() {
    // Clear buffer automatically.
  };

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_id = ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID);
  const auto &device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  device::ResKey res_key{device::GetDeviceTypeByName(device_name), device_id};
  auto res_manager = device::HalResManager::GetInstance().GetOrCreateResManager(res_key);
  MS_EXCEPTION_IF_NULL(res_manager);
  res_manager->BindDeviceToCurrentThread(false);
  auto callback_ret = res_manager->LaunchCallback(callback_func, kDefaultStreamIndex);
  if (!callback_ret) {
    MS_LOG(EXCEPTION) << "Async Copy memory launch callback failed";
  }
}

void GraphParameterStore::ReleaseData() {
  ProfilerRecorder profiler(ProfilerModule::kRuntime, ProfilerEvent::kReleaseResource, "GraphParameterStore");
  // Add copy data callback to avoid release data before async copy finished.
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  AddCopyDataCallBack(tensor_data_in_callback_);
  tensor_data_in_callback_.clear();

  for (auto index : non_weight_ref_max_inputs_) {
    std::pair<size_t, size_t> position{index.first, index.second};
    auto &kernel_tensor_with_info = parameter_kernel_tensors_[index.first][index.second];
    auto &kernel_tensor = kernel_tensor_with_info.first;
    if (kernel_tensor != nullptr) {
      auto &device_tensor = kernel_tensor->device_address();
      if (device_tensor != nullptr && device_tensor->new_ref_count() == SIZE_MAX &&
          !device_tensor->is_ptr_persisted()) {
        MS_LOG(DEBUG) << "Set store device tensor: " << device_tensor.get() << " ptr null, outer idx: " << index.first
                      << ", inner idx: " << index.second << ", info: " << kernel_tensor->ToString();
        release_data_info_[{position}] = {kernel_tensor->GetType(), device_tensor->GetNodeIndex()};
        kernel_tensor->set_device_address(nullptr);
      }
    }
  }

  for (auto &buffer : buffers_) {
    buffer.clear();
  }
  buffers_.clear();

  input_args_ = nullptr;
}

void GraphParameterStore::FillBuffer(size_t idx, const std::vector<TensorPtr> &tensors) {
  if (idx >= (*input_args_).size()) {
    MS_LOG(EXCEPTION) << "Index is out of buffer range.";
  }
  if (buffers_[idx].size() > 0) {
    return;
  }
  buffers_[idx] = tensors;
}

std::pair<bool, std::pair<TypePtr, KernelWithIndex>> GraphParameterStore::GetReleasePositionInfo(
  const DeviceTensorPosition &position) {
  std::shared_lock<std::shared_mutex> lock(param_mutex_);
  const auto &iter = release_data_info_.find({position});
  if (iter == release_data_info_.end()) {
    MS_LOG(INFO) << "Can not find type in store, where outer index: " << position.first
                 << ", inner index: " << position.second;
    return std::make_pair(false, std::make_pair(nullptr, std::make_pair(nullptr, 0)));
  }
  return std::make_pair(true, iter->second);
}

Tensor *GraphParameterStore::FlattenInputTensorByArg(size_t arg_index, const KernelWithIndex &front_node) {
  if (arg_index >= (*input_args_).size()) {
    MS_LOG(INFO) << "Arg index out of args range, index is " << arg_index << " and args size is "
                 << (*input_args_).size();
    return nullptr;
  }
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  std::vector<tensor::TensorPtr> flatten_tensors;
  tensor::TensorPtr tensor = nullptr;
  if (GetPositionTensor(arg_index)) {
    tensor = utils::cast<tensor::TensorPtr>((*input_args_)[arg_index]);
    MS_EXCEPTION_IF_NULL(tensor);
    flatten_tensors.emplace_back(tensor);
  } else {
    AnfAlgo::FlattenInputArg((*input_args_)[arg_index], front_node.first, &flatten_tensors);
    auto input_tensor_index = FetchInputTensorIndex(front_node);
    if (input_tensor_index >= flatten_tensors.size()) {
      MS_LOG(INFO) << "Input tensor index out of args range, index is " << input_tensor_index << " and tensors size is "
                   << flatten_tensors.size();
      return nullptr;
    }
    tensor = flatten_tensors[input_tensor_index];
  }
  // Return if already push into buffers.
  if (buffers_[arg_index].size() > 0) {
    return tensor.get();
  }
  buffers_[arg_index] = flatten_tensors;

  return tensor.get();
}

void GraphParameterStore::Clear() {
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  parameter_kernel_tensors_.clear();
  release_data_info_.clear();
  front_node_to_index_.clear();
  node_to_real_front_node_.clear();
  index_to_front_node_.clear();
  tensor_data_in_callback_.clear();
  for (auto &buffer : buffers_) {
    buffer.clear();
  }
  buffers_.clear();
}
}  // namespace runtime
}  // namespace mindspore
