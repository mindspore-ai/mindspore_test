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
    MS_LOG(EXCEPTION) << "Outer index is larger than the size of is weights [" << is_weights_.size() << "].";
  }
  return is_weights_[outer_index];
}

size_t GraphParameterStore::GetNonWeightParameterNum() { return is_weights_.size() - weight_num_; }

void GraphParameterStore::InsertTensorDataIntoCallback(const TensorDataPtr &tensor_data) {
  tensor_data_in_callback_.push_back(tensor_data);
}

void GraphParameterStore::InsertDeviceTensorIntoCallback(const DeviceTensorPtr &device_tensor) {
  device_tensor_in_callback_.push_back(device_tensor);
}

void GraphParameterStore::ResetPrepareState() {
  for (size_t i = 0; i < parameter_kernel_tensors_.size(); ++i) {
    auto &kernel_tensors = parameter_kernel_tensors_[i];
    for (size_t j = 0; j < kernel_tensors.size(); ++j) {
      kernel_tensors[j].second.second = false;
    }
  }
  tensor_data_in_callback_.reserve(buffer_size_);
  device_tensor_in_callback_.reserve(buffer_size_);
}

void GraphParameterStore::ResetAddrRefCount(size_t outer_index, size_t inner_index, DeviceTensorType value_type) {
  CheckIndexValid(outer_index, inner_index);
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
  auto &heter_kernel_tensor_with_info = heter_kernel_tensors_[outer_index][inner_index];
  bool is_ref_count_max =
    (kernel_tensor_with_info.second.first == SIZE_MAX || heter_kernel_tensor_with_info.second == SIZE_MAX);

  if (kernel_tensor_with_info.first != nullptr) {
    auto &device_tensor = kernel_tensor_with_info.first->device_address();
    if (device_tensor != nullptr && device_tensor->GetDeviceType() == value_type) {
      auto user_cnt = kernel_tensor_with_info.second.first;
      device_tensor->set_original_ref_count(user_cnt);
      device_tensor->ResetRefCount();
      if (user_cnt > 0) {
        // When allocate memory, the ref count would be increase, so it should be decrease here.
        if (is_ref_count_max) {
          device_tensor->set_new_ref_count(SIZE_MAX);
        } else {
          static std::string name = "Parameter store";
          device_tensor->IncreaseNewRefCount(name, user_cnt - 1);
        }
        device_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
        MS_LOG(DEBUG) << "Parameter store set new ref count:" << user_cnt - 1
                      << " for device address:" << device_tensor->PrintInfo();
      } else {
        MS_LOG(DEBUG) << "User count:0 for parameter store outer index:" << outer_index
                      << " inner index:" << inner_index << " for device address:" << device_tensor;
      }
      return;
    }
  }

  if (heter_kernel_tensor_with_info.first != nullptr) {
    auto &heter_device_tensor = heter_kernel_tensor_with_info.first->device_address();
    if (heter_device_tensor != nullptr && heter_device_tensor->GetDeviceType() == value_type) {
      auto user_cnt = heter_kernel_tensor_with_info.second;
      heter_device_tensor->set_original_ref_count(user_cnt);
      heter_device_tensor->ResetRefCount();
      if (user_cnt > 0) {
        // When allocate memory, the ref count would be increase, so it should be decrease here.
        if (is_ref_count_max) {
          heter_device_tensor->set_new_ref_count(SIZE_MAX);
        } else {
          static std::string name = "Parameter store";
          heter_device_tensor->IncreaseNewRefCount(name, user_cnt - 1);
        }
        heter_device_tensor->ClearFlag(device::kDeviceAddressFlagNotUsed);
        MS_LOG(DEBUG) << "Parameter store set new ref count:" << user_cnt - 1
                      << " for device address:" << heter_device_tensor->PrintInfo();
      } else {
        MS_LOG(DEBUG) << "User count:0 for parameter store outer index:" << outer_index
                      << " inner index:" << inner_index << " for device address:" << heter_device_tensor;
      }
    }
  }
}

KernelTensorPtr GraphParameterStore::FetchWithFreshRefMap(size_t outer_index, size_t inner_index,
                                                          DeviceTensorType value_type) {
  CheckIndexValid(outer_index, inner_index);
  std::shared_lock<std::shared_mutex> lock(param_mutex_);
  const auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
  const auto &kernel_tensor = kernel_tensor_with_info.first;
  const auto &heter_kernel_tensor_with_info = heter_kernel_tensors_[outer_index][inner_index];
  const auto &heter_kernel_tensor = heter_kernel_tensor_with_info.first;

  // Record non weight parameter ref map.
  if (kernel_tensor != nullptr && heter_kernel_tensor != nullptr) {
    const auto &iter = index_to_front_node_.find(outer_index);
    if (iter != index_to_front_node_.end() && iter->second->isa<Parameter>() &&
        !common::AnfAlgo::IsParameterWeight(iter->second->cast<ParameterPtr>())) {
      const auto &device_tensor = kernel_tensor->device_address();
      const auto &heter_device_tensor = heter_kernel_tensor->device_address();
      if (device_tensor != nullptr && heter_device_tensor != nullptr) {
        DeviceTensorCopyStore::GetInstance().Insert(device_tensor.get(), heter_device_tensor.get());
      }
    }
  }

  if (kernel_tensor != nullptr && device::GetDeviceTypeByName(kernel_tensor->device_name()) == value_type) {
    return kernel_tensor;
  }

  if (heter_kernel_tensor != nullptr && device::GetDeviceTypeByName(heter_kernel_tensor->device_name()) == value_type) {
    return heter_kernel_tensor;
  }

  // The parameter and actor input is heterogeneous, kernel actor will use copy input kernel tensor.
  if (heter_kernel_tensor == nullptr && kernel_tensor != nullptr) {
    return kernel_tensor;
  }
  return nullptr;
}

KernelTensorPtr GraphParameterStore::Fetch(size_t outer_index, size_t inner_index, DeviceTensorType value_type) {
  CheckIndexValid(outer_index, inner_index);
  std::shared_lock<std::shared_mutex> lock(param_mutex_);
  const auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
  const auto &kernel_tensor = kernel_tensor_with_info.first;
  if (kernel_tensor != nullptr && kernel_tensor->device_address() != nullptr &&
      device::GetDeviceTypeByName(kernel_tensor->device_address()->device_name()) == value_type) {
    return kernel_tensor;
  }

  const auto &heter_kernel_tensor_with_info = heter_kernel_tensors_[outer_index][inner_index];
  const auto &heter_kernel_tensor = heter_kernel_tensor_with_info.first;
  if (heter_kernel_tensor != nullptr && heter_kernel_tensor->device_address() != nullptr &&
      device::GetDeviceTypeByName(heter_kernel_tensor->device_address()->device_name()) == value_type) {
    return heter_kernel_tensor;
  }

  // The parameter and actor input is heterogeneous, kernel actor will use copy input kernel tensor.
  return kernel_tensor;
}

std::vector<KernelTensorPtr> GraphParameterStore::Fetch(size_t outer_index, size_t inner_index) {
  CheckIndexValid(outer_index, inner_index);
  std::shared_lock<std::shared_mutex> lock(param_mutex_);
  std::vector<KernelTensorPtr> input_list;
  const auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
  const auto &kernel_tensor = kernel_tensor_with_info.first;
  if (kernel_tensor != nullptr && kernel_tensor->device_address() != nullptr) {
    input_list.push_back(kernel_tensor);
  }

  const auto &heter_kernel_tensor_with_info = heter_kernel_tensors_[outer_index][inner_index];
  const auto &heter_kernel_tensor = heter_kernel_tensor_with_info.first;
  if (heter_kernel_tensor != nullptr && heter_kernel_tensor->device_address() != nullptr) {
    input_list.push_back(heter_kernel_tensor);
  }
  return input_list;
}

bool GraphParameterStore::HasHeter(size_t outer_index, size_t inner_index) {
  CheckIndexValid(outer_index, inner_index);
  std::shared_lock<std::shared_mutex> lock(param_mutex_);
  const auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
  const auto &kernel_tensor = kernel_tensor_with_info.first;
  const auto &heter_kernel_tensor_with_info = heter_kernel_tensors_[outer_index][inner_index];
  const auto &heter_kernel_tensor = heter_kernel_tensor_with_info.first;
  return kernel_tensor != nullptr && kernel_tensor->device_address() != nullptr && heter_kernel_tensor != nullptr &&
         heter_kernel_tensor->device_address() != nullptr;
}

void GraphParameterStore::Push(size_t outer_index, size_t inner_index, const KernelTensorPtr &value,
                               DeviceTensorType value_type, size_t cnt) {
  auto is_heter = CheckDeviceTensorHeter(outer_index, inner_index, value_type);
  std::unique_lock<std::shared_mutex> lock(param_mutex_);
  if (!is_heter) {
    auto &kernel_tensor_with_info = parameter_kernel_tensors_[outer_index][inner_index];
    kernel_tensor_with_info.first = value;
    kernel_tensor_with_info.second.first = cnt;
    return;
  }

  auto &heter_kernel_tensor_with_info = heter_kernel_tensors_[outer_index][inner_index];
  heter_kernel_tensor_with_info.first = value;
  heter_kernel_tensor_with_info.second = cnt;
}

Tensor *GraphParameterStore::FetchTensor(size_t args_index, const KernelWithIndex &node) const {
  if (args_index >= buffers_.size()) {
    MS_LOG(EXCEPTION) << "Index " << args_index << " is out of buffers range " << buffers_.size() << ".";
  }
  TensorPtr tensor = nullptr;
  if (buffers_[args_index].size() > 0) {
    if (node.second >= buffers_[args_index].size()) {
      MS_LOG(EXCEPTION) << "Node position " << node.second << " is out of buffers position "
                        << buffers_[args_index].size() << " range.";
    }
    tensor = buffers_[args_index][node.second];
  } else {
    tensor = FetchInputTensorByArg(*input_args_, args_index, node);
  }
  MS_EXCEPTION_IF_NULL(tensor);
  return tensor.get();
}

bool GraphParameterStore::RecordGraphInputsAndIsDyn(const std::vector<size_t> &input_index,
                                                    const std::vector<ParameterPtr> &parameters) {
  bool isDyn = false;
  auto &llm_manager = LLMManager::GetInstance();
  for (size_t l = 0; l < input_index.size(); ++l) {
    auto i = input_index[l];
    auto origin_parameter = parameters[l];
    MS_EXCEPTION_IF_NULL(origin_parameter);
    auto buffer_inner_size = buffers_[i].size();
    // List tensor input do not compare shape.
    if (buffer_inner_size != 1) {
      continue;
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

void AddCopyDataCallBack(const std::vector<TensorDataPtr> &tensor_data_in_callback,
                         const std::vector<DeviceTensorPtr> &device_tensor_in_callback) {
  device::CallbackFunc callback_func = [tensor_data_in_callback, device_tensor_in_callback]() {
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
  AddCopyDataCallBack(tensor_data_in_callback_, device_tensor_in_callback_);
  tensor_data_in_callback_.clear();
  device_tensor_in_callback_.clear();

  for (auto index : non_weight_ref_max_inputs_) {
    CheckIndexValid(index.first, index.second);
    std::pair<size_t, size_t> position{index.first, index.second};
    auto &kernel_tensor_with_info = parameter_kernel_tensors_[index.first][index.second];
    auto &kernel_tensor = kernel_tensor_with_info.first;
    if (kernel_tensor != nullptr) {
      auto &device_tensor = kernel_tensor->device_address();
      if (device_tensor != nullptr && device_tensor->original_ref_count() == SIZE_MAX &&
          !device_tensor->is_ptr_persisted()) {
        MS_LOG(DEBUG) << "Set store device tensor: " << device_tensor.get() << " ptr null, outer idx: " << index.first
                      << ", inner idx: " << index.second << ", info: " << device_tensor->PrintInfo();
        release_data_info_[{position, device_tensor->GetDeviceType()}] = {kernel_tensor->GetType(),
                                                                          device_tensor->GetNodeIndex()};
        kernel_tensor->set_device_address(nullptr);
      }
    }

    auto &heter_kernel_tensor_with_info = heter_kernel_tensors_[index.first][index.second];
    auto &heter_kernel_tensor = heter_kernel_tensor_with_info.first;
    if (heter_kernel_tensor != nullptr) {
      auto &heter_device_tensor = heter_kernel_tensor->device_address();
      if (heter_device_tensor != nullptr && heter_device_tensor->original_ref_count() == SIZE_MAX &&
          !heter_device_tensor->is_ptr_persisted()) {
        MS_LOG(DEBUG) << "Set store heter device tensor: " << heter_device_tensor.get()
                      << " ptr null, outer idx: " << index.first << ", inner idx: " << index.second
                      << ", info: " << heter_device_tensor->PrintInfo();
        release_data_info_[{position, heter_device_tensor->GetDeviceType()}] = {heter_kernel_tensor->GetType(),
                                                                                heter_device_tensor->GetNodeIndex()};
        heter_kernel_tensor->set_device_address(nullptr);
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

void GraphParameterStore::InsertRefDeviceTensors(const DeviceTensorPosition &key, DeviceTensor *value) {
  const auto &iter = ref_device_tensors_.find(key);
  if (iter == ref_device_tensors_.end()) {
    ref_device_tensors_[key] = {value};
    return;
  }
  ref_device_tensors_[key].insert(value);
}

std::pair<bool, std::pair<TypePtr, KernelWithIndex>> GraphParameterStore::GetReleasePositionInfo(
  const std::pair<size_t, size_t> &position, DeviceTensorType type) {
  const auto &iter = release_data_info_.find({position, type});
  if (iter == release_data_info_.end()) {
    MS_LOG(INFO) << "Can not find type in store, where outer index: " << position.first
                 << ", inner index: " << position.second << ", type: " << type;
    return std::make_pair(false, std::make_pair(nullptr, std::make_pair(nullptr, 0)));
  }
  return std::make_pair(true, iter->second);
}
}  // namespace runtime
}  // namespace mindspore
