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

#include "frontend/parallel/ops_info/moe_gating_top_k_softmax_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
Status MoeGatingTopKSoftmaxInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  auto input_strategys = strategy->GetInputDim();
  if (std::any_of(input_strategys.begin(), input_strategys.end(), [](const auto &input_stra) {
        return std::any_of(input_stra.begin(), input_stra.end(), [](const auto &dim_stra) { return dim_stra != 1; });
      })) {
    MS_LOG(ERROR) << "Invalid strategy for MoeGatingTopKSoftmax, shared num is now only supported 1.";
    return FAILED;
  }
  return SUCCESS;
}

Status MoeGatingTopKSoftmaxInfo::InferDevMatrixShape() {
  dev_matrix_shape_ = {};
  return SUCCESS;
}

Status MoeGatingTopKSoftmaxInfo::InferTensorMap() {
  for (size_t i = 0; i < inputs_shape_.size(); i++) {
    Shape tensor_map_index = {};
    for (size_t j = 0; j < inputs_shape_[i].size(); ++j) {
      tensor_map_index.push_back(MAP_NONE);
    }
    inputs_tensor_map_.push_back(tensor_map_index);
  }
  for (size_t i = 0; i < outputs_shape_.size(); i++) {
    Shape tensor_map_index;
    for (size_t j = 0; j < outputs_shape_[i].size(); ++j) {
      tensor_map_index.push_back(MAP_NONE);
    }
    outputs_tensor_map_.push_back(tensor_map_index);
  }
  return SUCCESS;
}

Status MoeGatingTopKSoftmaxInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map is empty";
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}

REGISTER(MoeGatingTopKSoftmaxInfo);
}  // namespace parallel
}  // namespace mindspore
