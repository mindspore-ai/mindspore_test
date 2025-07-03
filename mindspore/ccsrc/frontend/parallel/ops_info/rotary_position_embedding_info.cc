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

#include "frontend/parallel/ops_info/rotary_position_embedding_info.h"
#include <utility>
#include <algorithm>
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
Status RotaryPositionEmbeddingInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  // The strategy for each input tensor must be equal
  Strategies strategies = strategy->GetInputDim();
  auto strategy_size = strategies.size();
  auto strategy_item_size = strategies[0].size();
  for (size_t i = 0; i < strategy_item_size; ++i) {
    for (size_t j = 1; j < strategy_size; ++j) {
      if (((strategies[j][i] == 1 && inputs_shape_[j][i] == 1) || strategies[j][i] == strategies[0][i]) &&
          strategies[j][strategy_item_size - 1] == 1) {
        continue;
      } else {
        MS_LOG(ERROR) << name_ << ": The strategy for each input must be equal to strategies[0]: " << strategies[0]
                      << ", but got strategies[" << j << "]: " << strategies[j];
        return FAILED;
      }
    }
  }
  return SUCCESS;
}

Status RotaryPositionEmbeddingInfo::InferDevMatrixShape() {
  dev_matrix_shape_.clear();

  Strategies strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    return SUCCESS;
  }
  dev_matrix_shape_.assign(strategies[0].begin(), strategies[0].end());

  MS_LOG(INFO) << name_ << ": dev matrix: " << ShapeToString(dev_matrix_shape_);
  return SUCCESS;
}

Status RotaryPositionEmbeddingInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  Shape outputs_tensor_map;

  Strategies strategies = strategy_->GetInputDim();
  size_t dim = strategies.at(0).size();
  for (size_t i = 0; i < dim; ++i) {
    if (strategies[0][i] == 1) {
      outputs_tensor_map.push_back(-1);
    } else {
      outputs_tensor_map.push_back(dim - i - 1);
    }
  }
  (void)outputs_tensor_map_.emplace_back(std::move(outputs_tensor_map));

  for (size_t j = 0; j < strategies.size(); ++j) {
    Shape sub_tensor_map;
    for (size_t i = 0; i < dim; ++i) {
      if (strategies[j][i] == 1) {
        sub_tensor_map.push_back(-1);
      } else {
        sub_tensor_map.push_back(dim - i - 1);
      }
    }
    inputs_tensor_map_.push_back(sub_tensor_map);
  }

  return SUCCESS;
}

std::vector<StrategyPtr> RotaryPositionEmbeddingInfo::GenerateOpStrategies(int64_t stage_id) {
  Shapes splittable_inputs;
  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    (void)splittable_inputs.emplace_back(inputs_shape_[i].size());
    for (size_t j = 0; j < inputs_shape_[i].size(); ++j) {
      splittable_inputs[i][j] = SizeToLong(j) + 1;
    }
  }

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForDependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Generate strategies for dependent inputs() failed.";
  }

  return sp_vector;
}

void RotaryPositionEmbeddingInfo::ReComputeBatchSplitFlagList() {
  bool flag = false;
  if (!inputs_shape_[0].empty()) {
    flag = true;
  }

  // Batch dim of each input can be split
  for (size_t i = 0; i < split_flag_list_.size(); ++i) {
    split_flag_list_[i] = flag;
  }
}

REGISTER(RotaryPositionEmbeddingInfo);
}  // namespace parallel
}  // namespace mindspore
