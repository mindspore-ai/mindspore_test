/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/loss_info.h"

#include <algorithm>
#include <utility>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace parallel {
Status SoftmaxCrossEntropyWithLogitsInfo::CheckStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  Dimensions label_strategy = stra.at(1);
  if (input_strategy != label_strategy) {
    MS_LOG(ERROR) << name_ << " : Strategies of relevant dimensions are not equal.";
    return FAILED;
  }

  int64_t axis_index = axis_;
  if (axis_ < 0) {
    size_t input_dim = inputs_shape_.at(0).size();
    axis_index = static_cast<int64_t>(input_dim) + axis_;
  }

  int64_t input_axis_strategy = input_strategy.at(LongToSize(axis_index));
  int64_t label_axis_strategy = label_strategy.at(LongToSize(axis_index));
  // Dimension corresponding to axis is un-splittable
  if ((input_axis_strategy != MIN_SLICE_NUM) && (label_axis_strategy != MIN_SLICE_NUM)) {
    MS_LOG(ERROR) << name_ << " : The strategy corresponding to axis dimension is not 1, input: " << input_axis_strategy
                  << ", label: " << label_axis_strategy;
    return FAILED;
  }

  return SUCCESS;
}

Status SoftmaxCrossEntropyWithLogitsInfo::GetAttrs() {
  if ((inputs_shape_.size() != SoftmaxCrossEntropyWithLogitsInputsSize) ||
      (outputs_shape_.size() != SoftmaxCrossEntropyWithLogitsOutputsSize)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size or outputs shape size is wrong.";
    return FAILED;
  }
  return SUCCESS;
}

Status SoftmaxCrossEntropyWithLogitsInfo::InferDevMatrixShape() {
  Strategies stra = strategy_->GetInputDim();
  Dimensions input_strategy = stra.at(0);
  dev_matrix_shape_ = input_strategy;
  return SUCCESS;
}

Status SoftmaxCrossEntropyWithLogitsInfo::InferTensorMap() {
  Shape tensor_map_index;
  size_t size = inputs_shape_[0].size();
  // such as 4: tensor_map_index [3,2,1,0]
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back(static_cast<int64_t>(size - i - 1));
  }

  Shape first_output_tensor_map = {tensor_map_index[0]};
  inputs_tensor_map_.push_back(tensor_map_index);          // input
  inputs_tensor_map_.push_back(tensor_map_index);          // label
  outputs_tensor_map_.push_back(first_output_tensor_map);  // output-0
  outputs_tensor_map_.push_back(tensor_map_index);         // output-1
  return SUCCESS;
}

// There are two outputs for SoftmaxCrossEntropyWithLogits, and outputs[1] is used for grad and overload the function.
Status SoftmaxCrossEntropyWithLogitsInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.size() != 2) {
    MS_LOG(ERROR) << name_ << " : The size of outputs tensor map " << outputs_tensor_map_.size() << " is error.";
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[1]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_[1]) << ", as_loss_divisor_ is "
               << as_loss_divisor_;
  return SUCCESS;
}

void SoftmaxCrossEntropyWithLogitsInfo::ReComputeBatchSplitFlagList() {
  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    split_flag_list_[i] = true;
  }
}

std::vector<StrategyPtr> SoftmaxCrossEntropyWithLogitsInfo::GenerateOpStrategies(int64_t stage_id) {
  int64_t axis_index = axis_;
  if (axis_ < 0) {
    size_t input_dim = inputs_shape_[0].size();
    axis_index = static_cast<int64_t>(input_dim) + axis_;
  }

  Shape input0_split;
  (void)input0_split.insert(input0_split.cbegin(), inputs_shape_[0].size(), 1);
  input0_split[LongToSize(axis_index)] = 0;
  Shapes splittable_inputs = {input0_split, input0_split};
  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesWithBroadcast(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << " : Generate strategies failed.";
  }

  return sp_vector;
}

Status SoftmaxCrossEntropyWithLogitsInfo::SetCostUnderStrategy(const StrategyPtr &strategy) {
  return SetCostUnderStrategyBase(strategy);
}

Status CrossEntropyLossInfo::GetAttrs() {
  if ((inputs_shape_.size() != CROSS_ENTROPY_LOSS_INPUTS_SIZE) &&
      (inputs_shape_.size() != CROSS_ENTROPY_LOSS_INPUTS_WITH_WEIGHT_SIZE)) {
    MS_LOG(ERROR) << name_ << " : Inputs shape size or outputs shape size is wrong.";
    return FAILED;
  }

  if (outputs_shape_.size() != CROSS_ENTROPY_LOSS_OUTPUTS_SIZE) {
    MS_LOG(ERROR) << name_ << " : Outputs shape size is wrong.";
    return FAILED;
  }

  // outputs: Tuple -> {[N,], [N, C], [0], [0]}
  const auto origin_outputs_shape = outputs_shape_;
  outputs_shape_.clear();
  for (const auto &output_shape : origin_outputs_shape) {
    const auto out_it =
      std::find_if(output_shape.cbegin(), output_shape.cend(), [&](const int64_t ele) { return ele == 0; });
    if (out_it == output_shape.end()) {
      outputs_shape_.push_back(output_shape);
    }
  }

  // reduction
  const std::string &op_name = GetPrimNameFromInfoName(this->name_);
  const std::optional<int64_t> &reduction_opt_ = GetScalarValueFromInputs<int64_t>(input_value_, op_name, REDUCTION);
  if (reduction_opt_.has_value()) {
    const int64_t reduction_val = reduction_opt_.value();
    if ((reduction_val != Reduction::NONE) && (reduction_val != Reduction::MEAN) &&
        (reduction_val != Reduction::REDUCTION_SUM)) {
      MS_LOG(ERROR) << name_ << " : `reduction` is not one of `none`, `mean`, `sum`.";
      return FAILED;
    }
    reduction_ = reduction_val;
  }

  // ignore_index
  const std::optional<int64_t> &ignore_index_opt_ =
    GetScalarValueFromInputs<int64_t>(input_value_, op_name, IGNORE_INDEX);
  if (ignore_index_opt_.has_value()) {
    const int64_t ignore_index_val = ignore_index_opt_.value();
    ignore_index_ = ignore_index_val;
  }

  // label_smoothing
  const std::optional<float_t> &label_smoothing_opt_ =
    GetScalarValueFromInputs<float_t>(input_value_, op_name, LABEL_SMOOTHING);
  if (label_smoothing_opt_.has_value()) {
    const float_t label_smoothing_val = label_smoothing_opt_.value();
    if ((label_smoothing_val < 0.0) || (label_smoothing_val > 1.0)) {
      MS_LOG(ERROR) << name_ << " : 'label_smoothing' is " << label_smoothing_val << ", it is out of range [0.0, 1.0].";
      return FAILED;
    }
    label_smoothing_ = label_smoothing_val;
  }

  return SUCCESS;
}

Status CrossEntropyLossInfo::CheckStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  const Strategies stra = strategy->GetInputDim();
  const Dimensions &input_strategy = stra.at(0);
  const Dimensions &label_strategy = stra.at(1);

  // The inputs and label strategy at batch dimension should be same
  if (input_strategy[0] != label_strategy[0]) {
    MS_LOG(ERROR) << name_ << " : Strategies of batch dimensions are not equal.";
    return FAILED;
  }

  // The input-0 is [N, C], the last dimension can not be splited
  int64_t axis_index = axis_;
  if (axis_ < 0) {
    const size_t input_dim = inputs_shape_.at(0).size();
    axis_index = static_cast<int64_t>(input_dim) + axis_;  // batch, N
  }

  // Dimension corresponding to axis is un-splittable for input-0
  const int64_t input_axis_strategy = input_strategy.at(LongToSize(axis_index));
  if (input_axis_strategy != MIN_SLICE_NUM) {
    MS_LOG(ERROR) << name_ << " : The strategy of inputs corresponding to num_class dimension is not 1, input: "
                  << input_axis_strategy;
    return FAILED;
  }

  // Only check weight when it exists
  constexpr size_t weight_index = CROSS_ENTROPY_LOSS_INPUTS_WITH_WEIGHT_SIZE - 1;
  if (inputs_shape_.size() == CROSS_ENTROPY_LOSS_INPUTS_WITH_WEIGHT_SIZE) {
    if (inputs_shape_[weight_index].empty()) {
      MS_LOG(ERROR) << name_ << " : Weight input exists but its tensor shape is empty.";
      return FAILED;
    }

    if (stra.size() <= weight_index) {
      MS_LOG(ERROR) << name_ << " : Weight exists but in_strategy size is not enough.";
      return FAILED;
    }

    const Dimensions &weight_strategy = stra.at(weight_index);
    if (weight_strategy.size() != 1 || weight_strategy[0] != MIN_SLICE_NUM) {
      MS_LOG(ERROR) << name_ << " : The strategy for weight must be (1,), but got: " << weight_strategy;
      return FAILED;
    }
  }

  return SUCCESS;
}

Status CrossEntropyLossInfo::InferTensorMap() {
  inputs_tensor_map_.clear();
  outputs_tensor_map_.clear();

  Shape tensor_map_index;
  const size_t size = inputs_shape_[0].size();
  // such as 4: tensor_map_index [3,2,1,0], decide by max Tensor
  for (size_t i = 0; i < size; ++i) {
    tensor_map_index.push_back(static_cast<int64_t>(size - i - 1));
  }

  inputs_tensor_map_.push_back(tensor_map_index);       // input-0, input
  inputs_tensor_map_.push_back({tensor_map_index[0]});  // input-1, target

  // input-2, weight
  constexpr size_t weight_index = CROSS_ENTROPY_LOSS_INPUTS_WITH_WEIGHT_SIZE - 1;
  if (inputs_shape_.size() == CROSS_ENTROPY_LOSS_INPUTS_WITH_WEIGHT_SIZE && !inputs_shape_[weight_index].empty()) {
    Shape third_input_tensor_map = {tensor_map_index[1]};
    inputs_tensor_map_.push_back(third_input_tensor_map);
  }

  // output-0 is [N,] when reduction is none
  Shape first_output_tensor_map;
  if (reduction_ == Reduction::NONE) {
    first_output_tensor_map.push_back(tensor_map_index[0]);
  } else {
    first_output_tensor_map.push_back(-1);
  }

  outputs_tensor_map_.push_back(first_output_tensor_map);  // output-0, [N, ] or [1,]
  outputs_tensor_map_.push_back(tensor_map_index);         // output-1, [N, C]

  return SUCCESS;
}

void CrossEntropyLossInfo::ReComputeBatchSplitFlagList() {
  split_flag_list_.clear();
  split_flag_list_.resize(inputs_shape_.size(), true);  // input-0 and input-1 is splittable at batch dimensions

  bool has_weight = (inputs_shape_.size() == CROSS_ENTROPY_LOSS_INPUTS_WITH_WEIGHT_SIZE);
  if (has_weight) {
    const size_t weight_index = inputs_shape_.size() - 1;
    split_flag_list_[weight_index] = false;  // weight is un-splittable at batch dimensions
  }
}

std::vector<StrategyPtr> CrossEntropyLossInfo::GenerateOpStrategies(int64_t stage_id) {
  int64_t axis_index = axis_;
  if (axis_ < 0) {
    const size_t input_dim = inputs_shape_[0].size();
    axis_index = static_cast<int64_t>(input_dim) + axis_;
  }

  // input-0, input
  Shape input0_split(inputs_shape_[0].size(), 1);  // 1 represent splitable
  input0_split[LongToSize(axis_index)] = 0;        // 0 represent unspliable

  // input-1, target
  Shape input1_split(inputs_shape_[1].size(), 1);  // fully splittable

  Shapes splittable_inputs = {input0_split, input1_split};

  // input-2, weight (optional)
  constexpr size_t weight_index = CROSS_ENTROPY_LOSS_INPUTS_WITH_WEIGHT_SIZE - 1;
  if (inputs_shape_.size() == CROSS_ENTROPY_LOSS_INPUTS_WITH_WEIGHT_SIZE && !inputs_shape_[weight_index].empty()) {
    Shape input2_split(inputs_shape_[weight_index].size(), 0);  // weight must be un-splittable
    splittable_inputs.push_back(input2_split);
  }

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesWithBroadcast(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << " : Generate strategies failed.";
  }

  return sp_vector;
}

Status CrossEntropyLossInfo::InferGroup() {
  Shape group_create_map;
  if (repeated_calc_num_ > 1) {
    const int64_t index = repeated_num_in_dev_matrix_right_ ? 0 : (static_cast<int64_t>(dev_matrix_shape_.size()) - 1);
    group_create_map.push_back(index);
  }

  if (CreateGroupByTensorMap(group_create_map, &group_list_) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }
  return SUCCESS;
}

Status CrossEntropyLossInfo::InferForwardCommunication() {
  if (reduction_ == Reduction::NONE) {
    MS_LOG(DEBUG) << name_ << ": reduction is " << reduction_ << ", there is no need to append reduce op.";
    return SUCCESS;
  }

  if (InferGroup() != SUCCESS) {
    MS_LOG(ERROR) << "Infer group failed";
    return FAILED;
  }

  if (group_list_.empty() || group_list_.front().GetDevNum() <= kSizeOne) {
    MS_LOG(INFO) << name_ << ": Forward all reduce is not required";
    return SUCCESS;
  }

  forward_op_list_.clear();

  MS_EXCEPTION_IF_NULL(outputs_dtype_);  // Tuple[Tensor[Float32]*2]
  const auto tuple_type_ptr = outputs_dtype_->cast_ptr<Tuple>();
  MS_EXCEPTION_IF_NULL(tuple_type_ptr);
  const auto &tuple_elements = tuple_type_ptr->elements();
  if (tuple_elements.empty()) {
    MS_LOG(ERROR) << "The output of CrossEntropyLoss is none, so infer forward communication failed.";
    return FAILED;
  }

  if (reduction_ == Reduction::MEAN) {
    for (size_t i = 0; i < tuple_elements.size(); ++i) {
      auto tensor_type_ptr = tuple_elements[i]->cast<mindspore::TensorTypePtr>();  // Get each tensor type
      MS_EXCEPTION_IF_NULL(tensor_type_ptr);
      auto element_type = tensor_type_ptr->element();  // Get the element type of the tensor

      // insert {mean, div} operator for output-0
      ForwardOp mean_div_op_list;
      if (i == 0) {
        mean_div_op_list = CreateAllReduceMeanForwardOp(group_list_[0], element_type);
        forward_op_list_.push_back(mean_div_op_list);
        continue;
      }

      // insert empty op vector for output-1
      (void)forward_op_list_.emplace_back();
    }
  }

  if (reduction_ == Reduction::REDUCTION_SUM) {
    for (size_t i = 0; i < tuple_elements.size(); ++i) {
      auto tensor_type_ptr = tuple_elements[i]->cast<mindspore::TensorTypePtr>();
      MS_EXCEPTION_IF_NULL(tensor_type_ptr);
      auto element_type = tensor_type_ptr->element();

      // insert {sum} operator for output-0
      ForwardOp sum_op_list;
      if (i == 0) {
        Operator sum_op = CreateAllReduceOp(REDUCE_OP_SUM, group_list_[0].name());
        sum_op_list.push_back(sum_op);
        forward_op_list_.push_back(sum_op_list);
        continue;
      }

      // insert empty op vector for output-1
      (void)forward_op_list_.emplace_back();
    }
  }

  MS_LOG(INFO) << name_ << ": The group name of forward communication is " << group_list_[0].name();
  return SUCCESS;
}

REGISTER(SoftmaxCrossEntropyWithLogitsInfo);
REGISTER(CrossEntropyLossInfo);
}  // namespace parallel
}  // namespace mindspore
