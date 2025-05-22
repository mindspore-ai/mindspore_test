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

#include "frontend/parallel/ops_info/index_add_info.h"

#include <ostream>

#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/ops_info/elementary_function_info.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kNameDim = "dim";
constexpr auto kNameInputDim = 0;
constexpr auto kNameIndexDim = 1;
constexpr auto kNameSourceDim = 2;
}  // namespace

// The dim dimension of input and source can not be split.
// The indices can not be split.
// The shape of input:  [A, B, ..., D, ..., M], the strategy of input: (a, b,..., 1, ..., m)
// The shape of index:  [N], the strategy of indices: (1,)
// The shape of source: [A, B, ..., N, ..., M], the strategy of source: (a, b, ..., 1,  ..., m)
// The shape of output: [A, B, ..., D, ..., M], the strategy of output: (a, b, ..., 1,  ..., m)
// The dev matrix: (a, b, ..., 1, ..., m)

// Get the dim value
Status IndexAddExtInfo::GetAttrs() {
  auto dim_opt = GetScalarValueFromInputs<int64_t>(input_value_, name_, kNameDim);
  if (!dim_opt.has_value()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", failed to get the input value of parameter 'dim'.";
    return FAILED;
  }
  auto dim = dim_opt.value();
  dim_ = LongToSize(dim);
  MS_LOG(DEBUG) << "inputs_shape " << inputs_shape_;

  if (dim_ < 0 || dim_ >= inputs_shape_[kNameInputDim].size()) {
    MS_LOG(ERROR) << name_ << ": dim must be greater than or equal to 0 and less than input shape size, bug got dim_ "
                  << dim_ << " and inputs size " << inputs_shape_[kNameInputDim].size();
    return FAILED;
  }
  return SUCCESS;
}

// The strategy of tensor input and source must be equal
// Dimension corresponding to dim is un-splittable
Status IndexAddExtInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  // Check the strategy of input、index、source
  Strategies strategies = strategy->GetInputDim();
  if (strategies[kNameSourceDim] != strategies[kNameInputDim]) {
    MS_LOG(ERROR) << name_ << ": The strategy for Source must be equal to Input: " << strategies[kNameInputDim]
                  << ", but got Source strategies: " << strategies[kNameSourceDim];
    return FAILED;
  }

  for (size_t i = 0; i < strategies.size(); ++i) {
    if (i == kNameIndexDim) {
      int64_t dim_strategy = strategies[i].at(LongToSize(kIndex0));
      if (dim_strategy != MIN_SLICE_NUM) {
        MS_LOG(ERROR) << name_ << ": The index can not be split,  the strategy of which should be 1 but is "
                      << dim_strategy;
        return FAILED;
      }
      continue;
    }

    int64_t dim_strategy = strategies[i].at(LongToSize(dim_));
    if (dim_strategy != MIN_SLICE_NUM) {
      MS_LOG(ERROR) << name_ << ": The dimensions corresponding to dim can not be split, but the strategy is "
                    << dim_strategy;
      return FAILED;
    }
  }

  return SUCCESS;
}

Status IndexAddExtInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  std::vector<Dimensions> strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the shard strategies is empty.";
    return FAILED;
  }
  dev_matrix_shape_ = strategies[kIndex0];
  return SUCCESS;
}

Status IndexAddExtInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  TensorMap param_tensor_map;
  TensorMap index_tensor_map;
  TensorMap output_tensor_map;

  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the inputs shape is empty.";
    return FAILED;
  }

  size_t size = inputs_shape_[kIndex0].size();
  for (size_t i = 0; i < size; ++i) {
    param_tensor_map.push_back(SizeToLong(size - i - 1));
  }

  index_tensor_map.push_back(-1);

  inputs_tensor_map_.push_back(param_tensor_map);   // input
  inputs_tensor_map_.push_back(index_tensor_map);   // index
  inputs_tensor_map_.push_back(param_tensor_map);   // source
  outputs_tensor_map_.push_back(param_tensor_map);  // output
  return SUCCESS;
}

Status IndexAddExtInfo::InferTensorInfo() {
  if (inputs_shape_.empty() || outputs_shape_.empty() || inputs_tensor_map_.empty() || outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": Invalid args";
    return FAILED;
  }

  // handle the tensor input, index and source
  for (size_t i = 0; i < inputs_tensor_map_.size(); ++i) {
    TensorLayout input_layout;
    if (input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_[i], inputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer input tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo input_tensor_info(input_layout);
    inputs_tensor_info_.push_back(input_tensor_info);
  }

  // handle the scalar dim and alpha
  TensorInfo dim_tensor_info;
  TensorInfo alpha_tensor_info;
  (void)inputs_tensor_info_.insert(inputs_tensor_info_.cbegin() + 1, dim_tensor_info);
  inputs_tensor_info_.push_back(alpha_tensor_info);

  for (size_t i = 0; i < outputs_tensor_map_.size(); ++i) {
    TensorLayout output_layout;
    if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_[i], outputs_shape_[i]) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer output tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo output_tensor_info(output_layout);
    outputs_tensor_info_.push_back(output_tensor_info);
  }
  return SUCCESS;
}

Status IndexAddExtInfo::CheckInputLayout() {
  if (inputs_tensor_info_.size() != kSizeThree) {
    MS_LOG(ERROR) << "The size of input_tensor_layout for " << name_ << " is " << inputs_tensor_info_.size()
                  << " rather than 3.";
    return FAILED;
  }

  Strategies strategies;
  for (size_t i = 0; i < inputs_tensor_info_.size(); ++i) {
    strategies.push_back(inputs_tensor_info_[i].InferStrategy());
  }

  int64_t stage_id = 0;
  StrategyPtr strategy = std::make_shared<Strategy>(stage_id, strategies);

  return CheckStrategy(strategy);
}

Status IndexAddExtInfo::CheckOutputLayout() {
  if (outputs_tensor_info_.size() != kSizeOne) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_ << ", the size of outputs_tensor_info should be 1, but got "
                   << outputs_tensor_info_.size();
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", the size of outputs_tensor_info should be 1, but got "
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

Status IndexAddExtInfo::InferOutputTensorInfo() {
  TensorInfo dim_tensor_info;
  TensorInfo alpha_tensor_info;
  (void)inputs_tensor_info_.insert(inputs_tensor_info_.cbegin() + 1, dim_tensor_info);
  inputs_tensor_info_.push_back(alpha_tensor_info);

  auto output_infer_tensor_layout_ = inputs_tensor_info_[kIndex0].tensor_layout();
  TensorInfo output_tensor_info(output_infer_tensor_layout_);
  outputs_tensor_info_.push_back(output_tensor_info);
  is_infer_out_layout_ = true;
  return SUCCESS;
}

Status IndexAddExtInfo::SetCostUnderStrategy(const StrategyPtr &strategy) { return SetCostUnderStrategyBase(strategy); }

std::vector<StrategyPtr> IndexAddExtInfo::GenerateOpStrategies(int64_t stage_id) {
  // generate the first input's strategy
  Shape input_split(inputs_shape_[0].size(), 1);
  input_split[dim_] = 0;
  Shapes splittable_input = {input_split};
  Shapes tmp_inputs_shape = {inputs_shape_[0]};

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, splittable_input, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Generate strategies failed";
  }

  // generate the indices's strategy and the source's strategy
  for (auto &sp : sp_vector) {
    if ((sp == nullptr) || sp->GetInputDim().empty()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": The strategy is null or empty";
    }
    Strategies tmp_strategy;
    Dimensions first_input_strategy = sp->GetInputDim()[0];
    Dimensions indices_strategy(inputs_shape_[1].size(), 1);
    Dimensions source_strategy = first_input_strategy;

    tmp_strategy.push_back(first_input_strategy);  // input
    tmp_strategy.push_back(indices_strategy);      // indices, can not be split
    tmp_strategy.push_back(source_strategy);       // source
    sp->ResetInputs(tmp_strategy);
  }

  return sp_vector;
}

REGISTER(IndexAddExtInfo);
}  // namespace parallel
}  // namespace mindspore
