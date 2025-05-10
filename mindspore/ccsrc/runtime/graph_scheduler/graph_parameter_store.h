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

#ifndef MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_PARAMETER_STORE_H_
#define MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_PARAMETER_STORE_H_

#include <memory>
#include <map>
#include <set>
#include <vector>
#include <utility>
#include <shared_mutex>
#include "utils/ms_utils.h"
#include "include/backend/visible.h"
#include "common/device_address.h"
namespace mindspore {
namespace runtime {
using mindspore::tensor::Tensor;
using TensorPtr = std::shared_ptr<Tensor>;
using TensorDataPtr = std::shared_ptr<mindspore::tensor::TensorData>;
using DeviceTensor = mindspore::device::DeviceAddress;
using DeviceTensorType = mindspore::device::DeviceType;
using DeviceTensorPtr = std::shared_ptr<DeviceTensor>;
using KernelWithIndex = std::pair<AnfNodePtr, size_t>;
using UserCntWithPrepared = std::pair<size_t, bool>;
using DeviceTensorPosition = std::pair<std::pair<size_t, size_t>, DeviceTensorType>;
// The device tensor mainly includes address ptr, size and reference count,
// which represents the basic data structure of kernel launch and transfers between actors.
// The args are input from the front every step.
// The parameter device tensors (such as weight and non-weight parameter) and the args are save in
// the store, so they can be known by actors and be used for preparing data in actor.
class BACKEND_EXPORT GraphParameterStore {
 public:
  GraphParameterStore() = default;
  ~GraphParameterStore() = default;

  void Resize(size_t front_parameter_size) {
    parameter_device_tensors_.resize(front_parameter_size);
    heter_device_tensors_.resize(front_parameter_size);
    is_dynamic_.resize(front_parameter_size);
    is_weights_.resize(front_parameter_size, false);
  }
  bool HasHeter(size_t outer_index, size_t inner_index);
  void ResizePosition(size_t outer_index, size_t tuple_unfold_length) {
    if (outer_index >= parameter_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "inner index is larger than the size of parameter device tensors [" << outer_index << "].";
    }
    parameter_device_tensors_[outer_index].resize(tuple_unfold_length);
    heter_device_tensors_[outer_index].resize(tuple_unfold_length);
    is_dynamic_[outer_index].resize(tuple_unfold_length, false);
    buffer_size_ += tuple_unfold_length;
  }

  void CheckIndexValid(size_t outer_index, size_t inner_index) {
    if (outer_index >= parameter_device_tensors_.size()) {
      MS_LOG(EXCEPTION) << "Outer index is larger than the size of parameter device tensors ["
                        << parameter_device_tensors_.size() << "].";
    }
    if (inner_index >= parameter_device_tensors_[outer_index].size()) {
      MS_LOG(EXCEPTION) << "inner index is larger than the size of parameter device tensors ["
                        << parameter_device_tensors_[outer_index].size() << "].";
    }
  }

  bool CheckDeviceTensorHeter(size_t outer_index, size_t inner_index, DeviceTensorType value_type) {
    CheckIndexValid(outer_index, inner_index);
    auto &device_tensor_with_info = parameter_device_tensors_[outer_index][inner_index];
    auto &device_tensor = device_tensor_with_info.first;
    auto &heter_device_tensor_with_info = heter_device_tensors_[outer_index][inner_index];
    auto &heter_device_tensor = heter_device_tensor_with_info.first;
    if (device_tensor == nullptr && heter_device_tensor == nullptr) {
      return false;
    }
    if (device_tensor != nullptr && device_tensor->GetDeviceType() != value_type) {
      return true;
    }
    if (heter_device_tensor != nullptr && heter_device_tensor->GetDeviceType() == value_type) {
      return true;
    }
    return false;
  }

  void SetInputArgs(const VectorRef &args) {
    input_args_ = const_cast<VectorRef *>(&args);
    buffers_.resize(args.size());
    host_tensors_shape_.resize(args.size());
  }
  VectorRef *GetInputArgs() const { return input_args_; }

  void SetDeviceTensorPrepared(size_t outer_idx, size_t inner_idx, bool is_prepared) {
    CheckIndexValid(outer_idx, inner_idx);
    auto &device_tensor_with_info = parameter_device_tensors_[outer_idx][inner_idx];
    device_tensor_with_info.second.second = is_prepared;
  }
  bool GetDeviceTensorPrepared(size_t outer_idx, size_t inner_idx) {
    CheckIndexValid(outer_idx, inner_idx);
    auto &device_tensor_with_info = parameter_device_tensors_[outer_idx][inner_idx];
    return device_tensor_with_info.second.second;
  }

  void SetUserCnt(size_t outer_idx, size_t inner_idx, size_t cnt, DeviceTensorType value_type) {
    auto is_heter = CheckDeviceTensorHeter(outer_idx, inner_idx, value_type);
    if (!is_heter) {
      auto &device_tensor_with_info = parameter_device_tensors_[outer_idx][inner_idx];
      device_tensor_with_info.second.first = cnt;
      return;
    }
    auto &device_tensor_with_info = heter_device_tensors_[outer_idx][inner_idx];
    device_tensor_with_info.second = cnt;
  }

  void IncreaseUserCnt(size_t outer_idx, size_t inner_idx, DeviceTensorType value_type) {
    auto is_heter = CheckDeviceTensorHeter(outer_idx, inner_idx, value_type);
    if (!is_heter) {
      auto &device_tensor_with_info = parameter_device_tensors_[outer_idx][inner_idx];
      if (device_tensor_with_info.second.first != SIZE_MAX) {
        device_tensor_with_info.second.first++;
      }
      return;
    }
    auto &heter_device_tensor_with_info = heter_device_tensors_[outer_idx][inner_idx];
    if (heter_device_tensor_with_info.second != SIZE_MAX) {
      heter_device_tensor_with_info.second++;
    }
  }

  size_t GetUserCnt(size_t outer_idx, size_t inner_idx, DeviceTensorType value_type) {
    auto is_heter = CheckDeviceTensorHeter(outer_idx, inner_idx, value_type);
    if (!is_heter) {
      auto &device_tensor_with_info = parameter_device_tensors_[outer_idx][inner_idx];
      return device_tensor_with_info.second.first;
    }
    auto &device_tensor_with_info = heter_device_tensors_[outer_idx][inner_idx];
    return device_tensor_with_info.second;
  }

  void SetFrontNodeToIndex(AnfNode *node, size_t index) {
    MS_EXCEPTION_IF_NULL(node);
    const auto &iter = front_node_to_index_.find(node);
    if (iter != front_node_to_index_.end()) {
      MS_LOG(INFO) << "Update index for front node " << node->DebugString() << " in graph parameter store.";
      iter->second = index;
    }
    front_node_to_index_.emplace(node, index);
    index_to_front_node_.emplace(index, node);
  }
  size_t GetFrontNodeToIndex(AnfNode *node) {
    MS_EXCEPTION_IF_NULL(node);
    auto iter = front_node_to_index_.find(node);
    if (iter == front_node_to_index_.end()) {
      MS_LOG(EXCEPTION) << "Can not find index for front node " << node->DebugString() << " in graph parameter store.";
    }
    return iter->second;
  }

  void CorrectFrontNodeMap(const KernelWithIndex &node_with_index, const KernelWithIndex &real_node_with_index) {
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    MS_EXCEPTION_IF_NULL(real_node_with_index.first);
    const auto &iter = node_to_real_front_node_.find(node_with_index);
    if (iter != node_to_real_front_node_.end()) {
      MS_LOG(INFO) << "Node: " << node_with_index.first->DebugString() << ", index: " << node_with_index.second
                   << ", is already map to real front node: " << real_node_with_index.first->DebugString()
                   << ", index: " << real_node_with_index.second << " in graph parameter store.";
      return;
    }
    node_to_real_front_node_.emplace(node_with_index, real_node_with_index);
  }
  KernelWithIndex GetRealFrontNode(const KernelWithIndex &node_with_index) {
    MS_EXCEPTION_IF_NULL(node_with_index.first);
    auto iter = node_to_real_front_node_.find(node_with_index);
    if (iter == node_to_real_front_node_.end()) {
      MS_LOG(EXCEPTION) << "Can not find real front node for node " << node_with_index.first->DebugString()
                        << ", index: " << node_with_index.second << " in graph parameter store.";
    }
    return iter->second;
  }

  bool IsFrontNodeInStore(AnfNode *node) {
    auto iter = front_node_to_index_.find(node);
    if (iter == front_node_to_index_.end()) {
      return false;
    }
    return true;
  }

  void SetIsPositionDynamic(size_t outer_index, size_t inner_index, bool is_dynamic) {
    CheckIndexValid(outer_index, inner_index);
    is_dynamic_[outer_index][inner_index] = is_dynamic;
  }

  bool IsPositionDynamic(size_t outer_index, size_t inner_index) {
    CheckIndexValid(outer_index, inner_index);
    return is_dynamic_[outer_index][inner_index];
  }

  void InsertNonWeightRefMaxInputs(size_t outer_index, size_t inner_index) {
    CheckIndexValid(outer_index, inner_index);
    non_weight_ref_max_inputs_.emplace(outer_index, inner_index);
  }

  const std::set<std::pair<size_t, size_t>> &GetNonWeightRefMaxInputs() const { return non_weight_ref_max_inputs_; }

  // Reset the prepare state at the step beginning.
  void ResetPrepareState();

  void ResetAddrRefCount(size_t outer_index, size_t inner_index, DeviceTensorType value_type);

  // Fetch device tensor with index from parameter_device_tensors_.
  DeviceTensor *Fetch(size_t outer_index, size_t inner_index, DeviceTensorType value_type);
  DeviceTensorPtr FetchMutableAddr(size_t outer_index, size_t inner_index, DeviceTensorType value_type);

  std::vector<DeviceTensor *> Fetch(size_t outer_index, size_t inner_index);
  std::vector<DeviceTensorPtr> FetchMutableAddr(size_t outer_index, size_t inner_index);

  // Push the device tensor and user count to parameter_device_tensors_.
  void Push(size_t outer_index, size_t inner_index, const DeviceTensorPtr &value, DeviceTensorType value_type,
            size_t cnt);

  // Fetch Tensor with index from input_args_.
  Tensor *FetchTensor(size_t args_index, const KernelWithIndex &node) const;

  // Record graph inputs and return whether is dynamic.
  bool RecordGraphInputsAndIsDyn(const std::vector<size_t> &input_index, const std::vector<ParameterPtr> &parameters);

  // Release input data at the end of run graph.
  void ReleaseData();

  void SetPositionWeight(size_t outer_index, bool is_weight);
  bool GetPositionWeight(size_t outer_index);
  size_t GetNonWeightParameterNum();

  // Insert and refresh ref device tensor when device tensor changed in store.
  void InsertRefDeviceTensors(const DeviceTensorPosition &key, DeviceTensor *value);
  void RefreshRefDeviceTensor(const DeviceTensorPosition &key);

  // Insert host tensor data and src device tensor into callback to avoid release before async copy finished.
  void InsertTensorDataIntoCallback(const TensorDataPtr &tensor_data);
  void InsertDeviceTensorIntoCallback(const DeviceTensorPtr &device_tensor);

  void Clear() {
    std::unique_lock<std::shared_mutex> lock(param_mutex_);
    parameter_device_tensors_.clear();
    heter_device_tensors_.clear();
    release_data_info_.clear();
    front_node_to_index_.clear();
    node_to_real_front_node_.clear();
    index_to_front_node_.clear();
    ref_device_tensors_.clear();
    tensor_data_in_callback_.clear();
    device_tensor_in_callback_.clear();
    for (auto &buffer : buffers_) {
      buffer.clear();
    }
    buffers_.clear();
  }

  const std::vector<std::vector<std::pair<DeviceTensorPtr, UserCntWithPrepared>>> &GetAll() const {
    return parameter_device_tensors_;
  }

  const std::vector<std::vector<std::pair<DeviceTensorPtr, size_t>>> &GetAllHeter() const {
    return heter_device_tensors_;
  }

  void FillBuffer(size_t idx, const std::vector<TensorPtr> &tensors);

  std::pair<bool, std::pair<TypePtr, KernelWithIndex>> GetReleasePositionInfo(const std::pair<size_t, size_t> &position,
                                                                              DeviceTensorType type);

 private:
  // The input args refresh in every step.
  VectorRef *input_args_;
  // The device tensors used for launch and transfer between actors. Outer index corresponds to the
  // front nodle position, and inner index corresponds to the addr position after tuple unfold.
  // Besides, record the user cnt and data prepared flag for each device tensor.
  std::vector<std::vector<std::pair<DeviceTensorPtr, UserCntWithPrepared>>> parameter_device_tensors_;
  // Record the heterogeneous device tensor of parameter_device_tensors_.
  std::vector<std::vector<std::pair<DeviceTensorPtr, size_t>>> heter_device_tensors_;
  // Record non-weight ref max input, so that do not tranverse the store when releasing data.
  std::set<std::pair<size_t, size_t>> non_weight_ref_max_inputs_;
  std::map<DeviceTensorPosition, std::pair<TypePtr, KernelWithIndex>> release_data_info_;

  std::map<AnfNode *, size_t> front_node_to_index_;

  std::vector<bool> is_weights_;
  size_t weight_num_{0};

  // When front node to index failed, use the map to find real front node.
  std::map<KernelWithIndex, KernelWithIndex> node_to_real_front_node_;
  std::map<size_t, AnfNode *> index_to_front_node_;
  // Store tensor from args.
  std::vector<std::vector<TensorPtr>> buffers_;
  size_t buffer_size_{0};
  // Protect async copy finished before release.
  std::vector<TensorDataPtr> tensor_data_in_callback_;
  std::vector<DeviceTensorPtr> device_tensor_in_callback_;
  // Record the dynamic shape for each position.
  std::vector<std::vector<bool>> is_dynamic_;
  // Record the ref map of device tensor in store.
  std::map<DeviceTensorPosition, std::set<DeviceTensor *>> ref_device_tensors_;
  // Record the tensor shape for inference.
  std::vector<ShapeVector> host_tensors_shape_;
  // Read/Write lock for map.
  mutable std::shared_mutex param_mutex_;
};
using GraphParameterStorePtr = std::shared_ptr<GraphParameterStore>;
}  // namespace runtime
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_RUNTIME_FRAMEWORK_GRAPH_PARAMETER_STORE_H_
