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

#include "frontend/parallel/ops_info/tracev2_info.h"

#include <utility>
#include <algorithm>

#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {

void TraceV2Info::ReComputeBatchSplitFlagList() {
  if (input_split_.empty()) {
    if (GetAxisSplit() != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": failed to get dim.";
    }
  }
  split_flag_list_[0] = input_split_[0];
}

std::vector<StrategyPtr> TraceV2Info::GenerateOpStrategies(int64_t stage_id) {
  if (input_split_.empty()) {
    if (GetAxisSplit() != SUCCESS) {
      MS_LOG(EXCEPTION) << name_ << ": failed to get dim.";
    }
  }

  Shapes splittable_inputs = {input_split_};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG(EXCEPTION) << name_ << ": Generate strategies for independent inputs() failed.";
  }
  return sp_vector;
}

Status TraceV2Info::GetAttrs() {
  MS_LOG(DEBUG) << "TraceV2Info: GetAttrs() start.";
  return GetAxisSplit();
}

Status TraceV2Info::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy.";
    return FAILED;
  }

  size_t strategy_size = strategy->GetInputNumber();
  if (strategy_size > 1) {
    MS_LOG(ERROR) << name_ << ": Strategy size cannot be greater than 1.";
    return FAILED;
  }

  Shape input_strategy = strategy->GetInputDim().at(0);
  size_t strategy_len = input_strategy.size();
  if (strategy_len != input_split_.size()) {
    MS_LOG(ERROR) << name_ << ": input strategy size cannot match input size.";
    return FAILED;
  }

  int64_t split_num = 1;
  for (size_t i = 0; i < strategy_len; i++) {
    if (input_strategy[i] > 1 && input_split_[i] == false) {
      MS_LOG(ERROR) << name_ << ": Cannot split tensor on Non-batch dim.";
      return FAILED;
    }
    split_num *= input_strategy[i];
  }

  if (split_num > stage_device_size_) {
    MS_LOG(ERROR) << name_ << " The number of splits cannot be greater than the number of devices.";
    return FAILED;
  }

  return SUCCESS;
}

Status TraceV2Info::InferTensorMap() {
  size_t size = input_split_.size();
  Shape tensor_map_in(size, -1);
  Shape tensor_map_out(size - kIndex2, -1);
  int64_t tensor_map_index = 0;
  size_t skip_cnt = 0;
  for (size_t i = 0; i < size; i++) {
    size_t index_i = size - i - 1;
    size_t index_j = index_i + skip_cnt - 2;
    if (input_split_[index_i]) {
      tensor_map_in[index_i] = tensor_map_index;
      tensor_map_out[index_j] = tensor_map_index;
    } else {
      skip_cnt += 1;
    }
    tensor_map_index++;
  }
  MS_LOG(DEBUG) << name_ << ": TensorMap value = " << ListToString(tensor_map_in);
  (void)inputs_tensor_map_.emplace_back(std::move(tensor_map_in));
  (void)outputs_tensor_map_.emplace_back(std::move(tensor_map_out));
  return SUCCESS;
}

Status TraceV2Info::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  Strategies stra = strategy_->GetInputDim();
  if (stra.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy can not be empty.";
    return FAILED;
  }
  dev_matrix_shape_ = stra.at(0);
  return SUCCESS;
}

Status TraceV2Info::InferMirrorOps() {
  if (OperatorInfo::InferMirrorOps() != SUCCESS) {
    return FAILED;
  }

  if (mirror_ops_.empty()) {
    return SUCCESS;
  }

  size_t index_num = GetIndexNum();
  for (size_t i = 1; i < index_num; i++) {
    // Push empty mirror op for n, dim, norm
    (void)mirror_ops_.emplace_back(OperatorVector());
  }
  return SUCCESS;
}

Status TraceV2Info::GetAxisSplit() {
  if (input_value_.size() != GetIndexNum()) {
    MS_LOG(ERROR) << name_ << ": Invalid inputs size " << input_value_.size();
    return FAILED;
  }

  if (!input_value_[GetAxisIndex1()]->isa<Int64Imm>() || !input_value_[GetAxisIndex2()]->isa<Int64Imm>()) {
    MS_LOG(ERROR) << name_ << ": The type of axis is not int64_t";
    return FAILED;
  }

  int64_t axis1 = GetValue<int64_t>(input_value_[GetAxisIndex1()]);
  int64_t axis2 = GetValue<int64_t>(input_value_[GetAxisIndex2()]);
  int64_t input_dim = SizeToLong(inputs_shape_[0].size());
  if ((axis1 > (input_dim - 1)) || (axis1 < -input_dim) || (axis2 > (input_dim - 1)) || (axis2 < -input_dim)) {
    MS_LOG(ERROR) << name_ << ": The axis(" << axis1 << ") is out of range[" << (-input_dim) << ", " << (input_dim - 1)
                  << "]";
    return FAILED;
  }

  if (axis1 < 0) {
    axis1 += input_dim;
  }

  if (axis2 < 0) {
    axis2 += input_dim;
  }

  std::vector<int64_t> splits(inputs_shape_[0].size(), 1);
  splits[axis1] = 0;
  splits[axis2] = 0;
  input_split_.swap(splits);
  MS_LOG(DEBUG) << name_ << ": The input_split_ size is " << input_split_.size() << ", value is "
                << ListToString(input_split_);
  return SUCCESS;
}

REGISTER(TraceV2Info);
}  // namespace parallel
}  // namespace mindspore
