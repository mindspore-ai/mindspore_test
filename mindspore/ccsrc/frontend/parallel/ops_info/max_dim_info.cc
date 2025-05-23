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

#include <utility>
#include <algorithm>

#include "frontend/parallel/status.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/ops_info/max_dim_info.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kNameDim = "dim";
constexpr auto kNameKeepDim = "keepdim";
}  // namespace

Status MaxDimInfo::GetAttrs() {
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the inputs shape is empty.";
    return FAILED;
  }
  int rank = SizeToInt(inputs_shape_[kIndex0].size());

  // get attr dim
  auto dim_opt = GetScalarValueFromInputs<int64_t>(input_value_, name_, kNameDim);
  if (!dim_opt.has_value()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", failed to get the input value of parameter 'dim'.";
    return FAILED;
  }
  if (dim_opt >= rank || dim_opt < -rank) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the value of parameter 'dim' is out of range ["
                  << (-rank) << ", " << (rank - 1) << "], the 'dim' is " << dim_opt.value() << " and the input.size() "
                  << "is " << rank << ".";
    return FAILED;
  }
  auto dim = dim_opt.value() < 0 ? dim_opt.value() + rank : dim_opt.value();
  dim_ = LongToSize(dim);

  // get attr keepdim
  auto keepdim_opt = GetScalarValueFromInputs<bool>(input_value_, name_, kNameKeepDim);
  if (!keepdim_opt.has_value()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", failed to get the input value of parameter 'keepdim'.";
    return FAILED;
  }
  keepdim_ = keepdim_opt.value();

  return SUCCESS;
}

Status MaxDimInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  auto strategies = strategy->GetInputDim();
  auto input_strategy = strategies[kIndex0];

  if (dim_ >= input_strategy.size()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the value of parameter 'dim' is out of range, the 'dim'"
                  << " is " << dim_ << " and the input_strategy.size() is " << input_strategy.size() << ".";
    return FAILED;
  }

  if (input_strategy[dim_] != 1) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the input's dimension 'dim' can not be split, the 'dim'"
                  << " is " << dim_ << " and the shard strategy is " << input_strategy << ".";
    return FAILED;
  }

  return SUCCESS;
}

Status MaxDimInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the shard strategies is empty.";
    return FAILED;
  }

  dev_matrix_shape_ = strategies[kIndex0];
  return SUCCESS;
}

Status MaxDimInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  TensorMap input_tensor_map;
  TensorMap output_tensor_map;
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the inputs shape is empty.";
    return FAILED;
  }

  // cannot use dev_matrix_shape_ replace inputs_shape_[0], because it may not be fully split in all devices.
  size_t size = inputs_shape_[kIndex0].size();
  for (size_t i = 0; i < size; ++i) {
    input_tensor_map.push_back(SizeToLong(size - i - 1));
  }

  for (size_t i = 0; i < size; ++i) {
    if (i == dim_) {
      if (keepdim_) {
        output_tensor_map.push_back(MAP_NONE);
      } else {
        continue;
      }
    } else {
      output_tensor_map.push_back(input_tensor_map[i]);
    }
  }
  inputs_tensor_map_.push_back(input_tensor_map);    // input
  outputs_tensor_map_.push_back(output_tensor_map);  // values
  outputs_tensor_map_.push_back(output_tensor_map);  // index
  return SUCCESS;
}

Status MaxDimInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the outputs tensor map is empty.";
    return FAILED;
  }

  MS_LOG(DEBUG) << "For distributed operator " << name_ << ", it has two outputs, use output[0] to infer";
  if (outputs_tensor_map_[kIndex0].empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(DEBUG) << "For distributed operator " << name_ << ", The output is a scalar, use the dev size"
                  << as_loss_divisor_ << " as loss divisor";
    return SUCCESS;
  }

  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[kIndex0]);

  MS_LOG(DEBUG) << "For distributed operator " << name_ << ", the dev matrix is " << dev_matrix_shape_ << ", the output"
                << " tensor map is " << outputs_tensor_map_[kIndex0] << ", the loss divisor is " << as_loss_divisor_;
  return SUCCESS;
}

std::vector<StrategyPtr> MaxDimInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split;
  for (size_t i = 0; i < inputs_shape_[0].size(); ++i) {
    if (i == dim_) {
      input0_split.push_back(0);
    } else {
      input0_split.push_back(1);
    }
  }
  Shapes splittable_inputs = {input0_split};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For distributed operator " << name_ << ", generate strategies failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For distributed operator " << name_ << ", No available strategy";
  }

  return sp_vector;
}

Status MaxDimInfo::CheckInputLayout() {
  if (inputs_tensor_info_.size() != kSizeOne) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_ << ", the size of inputs_tensor_info should be 1, but got "
                   << inputs_tensor_info_.size() << ".";
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", the size of inputs_tensor_info should be 1, but got "
                    << inputs_tensor_info_.size() << ".";
    }
    return FAILED;
  }
  auto input_tensor_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto input_tensor_map = input_tensor_layout.tensor_map_before();
  dev_matrix_shape_ = input_tensor_layout.device_arrangement_origin().array();
  auto device_dim = SizeToLong(dev_matrix_shape_.size());
  Shapes input_shard_strategy;
  Shape dim_shard_strategy;
  for (size_t i = 0; i < input_tensor_map.size(); ++i) {
    dim_shard_strategy.clear();
    for (size_t j = 0; j < input_tensor_map[i].size(); ++j) {
      auto shard_idx = device_dim - 1 - input_tensor_map[i][j];
      dim_shard_strategy.push_back(dev_matrix_shape_[shard_idx]);
    }
    input_shard_strategy.push_back(dim_shard_strategy);
  }
  MS_LOG(DEBUG) << "For distributed operator " << name_ << ":" << std::endl
                << "the dev_matrix is " << dev_matrix_shape_ << std::endl
                << "the input tensor shape is " << input_tensor_layout.tensor_shape_before().array() << std::endl
                << "the input tensor shape after expanding multiple split dimensions is "
                << input_tensor_layout.tensor_shape_origin().array() << std::endl
                << "the input tensor map is " << input_tensor_map << std::endl
                << "the input tensor map after expanding multiple split dimensions is "
                << input_tensor_layout.origin_tensor_map().array() << std::endl
                << ", the input shard strategy corresponding to the input tensor map now is " << input_shard_strategy;
  if (input_shard_strategy[dim_].size() == 1 && input_shard_strategy[dim_][kIndex0] == 1) {
    return SUCCESS;
  } else {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_
                   << ", the input's dimension 'dim' can not be split, the 'dim'"
                   << " is " << dim_ << " and the input shard strategy is " << input_shard_strategy << ".";
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_
                    << ", the input's dimension 'dim' can not be split, the 'dim'"
                    << " is " << dim_ << " and the input shard strategy is " << input_shard_strategy << ".";
    }
    return FAILED;
  }
  return SUCCESS;
}

Status MaxDimInfo::InferOutputTensorInfo() {
  auto input_tensor_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto input_tensor_map = input_tensor_layout.tensor_map_before();
  Shapes outputs_tensor_map = {};
  for (size_t i = 0; i < input_tensor_map.size(); ++i) {
    if (i == dim_) {
      if (keepdim_) {
        outputs_tensor_map.push_back({-1});
      } else {
        continue;
      }
    } else {
      outputs_tensor_map.push_back(input_tensor_map[i]);
    }
  }
  if ((output_infer_tensor_layout_.InitFromExtendVector(dev_matrix_shape_, outputs_tensor_map,
                                                        outputs_shape_[kIndex0]) != SUCCESS)) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the output_tensor_layout init failed.";
    return FAILED;
  }
  if (output_infer_tensor_layout_.tensor_shape_before().array() != outputs_shape_[kIndex0]) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the infer output shape "
                  << output_infer_tensor_layout_.tensor_shape_before().array() << " does not match the output shape "
                  << outputs_shape_[kIndex0];
    return FAILED;
  }
  TensorInfo output_tensor_info(output_infer_tensor_layout_);
  outputs_tensor_info_.push_back(output_tensor_info);  // values
  outputs_tensor_info_.push_back(output_tensor_info);  // index
  is_infer_out_layout_ = true;
  return SUCCESS;
}

Status MaxDimInfo::CheckOutputLayout() {
  if (outputs_tensor_info_.size() != kSizeTwo) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_ << ", the size of outputs_tensor_info should be 2, but got "
                   << outputs_tensor_info_.size();
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", the size of outputs_tensor_info should be 2, but got "
                    << outputs_tensor_info_.size();
    }
    return FAILED;
  }
  if (!is_infer_out_layout_) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_
                   << ", the output tensor layout is not allowed to be set by users.";
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_
                    << ", the output tensor layout is not allowed to be set by users.";
    }
    return FAILED;
  }
  return SUCCESS;
}

Status MaxDimInfo::InferAsLossDivisorByLayout() {
  if (outputs_tensor_info_.size() != kSizeTwo) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the size of outputs_tensor_info should be 2, but got "
                  << outputs_tensor_info_.size();
    return FAILED;
  }

  TensorMaps outputs_tensor_map = outputs_tensor_info_[kIndex0].tensor_layout().tensor_map_before();
  if (outputs_tensor_map.empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(DEBUG) << "For distributed operator " << name_ << ": the output is a scalar, use the dev size "
                  << as_loss_divisor_ << " as loss divisor.";
    return SUCCESS;
  }

  auto out_dev_matrix_shape = outputs_tensor_info_[kIndex0].tensor_layout().device_arrangement_origin().array();
  if (out_dev_matrix_shape.empty()) {
    MS_LOG(DEBUG) << "For distributed operator " << name_ << ", out_dev_matrix_shape is empty";
    out_dev_matrix_shape = dev_matrix_shape_;
  }
  Shape squashed_tensor_map;
  for (const auto &tensor_map : outputs_tensor_map) {
    std::copy(tensor_map.begin(), tensor_map.end(), std::back_inserter(squashed_tensor_map));
  }

  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(out_dev_matrix_shape, squashed_tensor_map);
  MS_LOG(DEBUG) << "For distributed operator " << name_ << ", the dev matrix is " << out_dev_matrix_shape << ", the "
                << "output tensor map is " << squashed_tensor_map << ", the loss divisor is " << as_loss_divisor_;
  return SUCCESS;
}
REGISTER(MaxDimInfo);
}  // namespace parallel
}  // namespace mindspore
