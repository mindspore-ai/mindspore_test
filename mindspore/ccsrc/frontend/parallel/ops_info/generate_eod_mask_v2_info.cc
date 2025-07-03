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

#include "frontend/parallel/ops_info/generate_eod_mask_v2_info.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
Status GenerateEodMaskV2Info::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }
  return SUCCESS;
}

Status GenerateEodMaskV2Info::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions sub_strategy = stra.at(0);
  dev_matrix_shape_ = sub_strategy;
  return SUCCESS;
}

void GenerateEodMaskV2Info::ReComputeBatchSplitFlagList() {
  split_flag_list_[0] = true;
  split_flag_list_[1] = false;
}

Status GenerateEodMaskV2Info::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }
  if (mirror_ops_.empty()) {
    return SUCCESS;
  }

  OperatorVector op_for_axis;
  (void)mirror_ops_.emplace_back(std::move(op_for_axis));
  return SUCCESS;
}

Status GenerateEodMaskV2Info::InferTensorMap() {
  TensorMap sub_a_tensor_map;
  // TensorMap sub_b_tensor_map;
  Strategies stra = strategy_->GetInputDim();
  Dimensions sub_a_strategy = stra.at(0);
  size_t sub_a_strategy_size = sub_a_strategy.size();
  for (size_t i = 0; i < sub_a_strategy_size; ++i) {
    sub_a_tensor_map.push_back(static_cast<int64_t>(LAST_INDEX(sub_a_strategy_size) - i));
  }
  inputs_tensor_map_.push_back(sub_a_tensor_map);
  inputs_tensor_map_.push_back({});
  inputs_tensor_map_.push_back({});
  inputs_tensor_map_.push_back({});
  inputs_tensor_map_.push_back({});
  outputs_tensor_map_.push_back(sub_a_tensor_map);

  return SUCCESS;
}

Status GenerateEodMaskV2Info::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

REGISTER(GenerateEodMaskV2Info);
}  // namespace parallel
}  // namespace mindspore
