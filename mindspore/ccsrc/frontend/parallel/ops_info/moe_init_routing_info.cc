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

#include "frontend/parallel/ops_info/moe_init_routing_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
// MoeFinalizeRouting has 4 inputs and 3 outputs
// x:                       2D Tensor (num_row, h)
// rowIdx:                  2D Tensor (num_row, k)
// expertIdx:               2D Tensor (num_row, k)
// activeNum:               int64
// -------------------------------------------------
// expandedX:               2D Tensor (min(num_row, activeNum) * k, h)
// expandedRowIdx           1D Tensor (num_row * k,)
// expandedExpertIdx        1D Tensor (num_row * k,)

constexpr size_t InputNum = 3;
constexpr size_t OutputNum = 3;
constexpr size_t kX = 0;
constexpr size_t kRowIdx = 1;
constexpr size_t kExpertIdx = 2;

Status MoeInitRoutingInfo::GetInputNumsAndGetIdx(const StrategyPtr &strategy) {
  auto input_nums = strategy->GetInputDim().size();
  if (input_nums != InputNum) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: the input nums must be 3. But current input nums is " << input_nums;
    return FAILED;
  }
  return SUCCESS;
}

Status MoeInitRoutingInfo::CheckStrategy(const StrategyPtr &strategy) {
  auto input_strategys = strategy->GetInputDim();
  if (GetInputNumsAndGetIdx(strategy) != SUCCESS) {
    return FAILED;
  }
  auto strategy_x = input_strategys.at(kX);
  auto strategy_rowIdx = input_strategys.at(kRowIdx);
  auto strategy_expertIdx = input_strategys.at(kExpertIdx);

  if (strategy_x.at(0) != 1 && strategy_x.at(1) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The x can't be shard, but got"
                  << " shard num : (" << strategy_x.at(0) << ", " << strategy_x.at(1) << ")";
    return FAILED;
  }

  if (strategy_rowIdx.at(0) != 1 && strategy_rowIdx.at(1) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The rowIdx can't be shard, but got"
                  << " shard num : (" << strategy_rowIdx.at(0) << ", " << strategy_rowIdx.at(1) << ")";
    return FAILED;
  }

  if (strategy_expertIdx.at(0) != 1 && strategy_expertIdx.at(1) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The expertIdx can't be shard, but got"
                  << " shard num : (" << strategy_expertIdx.at(0) << ", " << strategy_expertIdx.at(1) << ")";
    return FAILED;
  }

  return SUCCESS;
}

Status MoeInitRoutingInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.size() != OutputNum) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map must be 3, but got " << outputs_tensor_map_.size();
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}

Status MoeInitRoutingInfo::InferDevMatrixShape() {
  dev_matrix_shape_ = {};
  return SUCCESS;
}

Status MoeInitRoutingInfo::InferTensorMap() {
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

REGISTER(MoeInitRoutingInfo);
}  // namespace parallel
}  // namespace mindspore
