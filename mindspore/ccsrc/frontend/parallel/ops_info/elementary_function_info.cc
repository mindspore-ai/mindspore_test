/**
 * Copyright 2022-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/elementary_function_info.h"

#include <ostream>

#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr int64_t kInvalidDimValue = -1;
}  // namespace
Status CholeskyInfo::GetAttrs() {
  axis_ = {-2, -1};
  return SUCCESS;
}

// the last two dimensions can not be split
Status CholeskyInfo::CheckStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy";
    return FAILED;
  }

  std::vector<Dimensions> stra = strategy->GetInputDim();
  Dimensions input_strategy = stra.at(0);

  for (auto &element : axis_) {
    int64_t axis_index = element;
    if (element < 0) {
      size_t input_dim = inputs_shape_.at(0).size();
      axis_index = static_cast<int64_t>(input_dim) + element;
    }

    int64_t axis_strategy = input_strategy.at(LongToSize(axis_index));
    // Dimension corresponding to axis is un-splittable
    if (axis_strategy != MIN_SLICE_NUM) {
      MS_LOG(ERROR) << name_ << ": The last two dimensions can not be split, but the strategy is " << input_strategy;
      return FAILED;
    }
  }

  return SUCCESS;
}

Status RepeatInterleaveInfo::GetAttrs() {
  if (input_value_.size() < kSizeFour) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the num of input should be 4, but got "
                  << input_value_.size();
    return FAILED;
  }
  if (inputs_shape_.empty()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the inputs shape is empty.";
    return FAILED;
  }
  const int64_t rank = SizeToLong(inputs_shape_[kIndex0].size());

  // get attr dim
  MS_EXCEPTION_IF_NULL(input_value_[kIndex2]);
  if (!input_value_[kIndex2]->isa<None>()) {
    auto dim_opt = GetScalarValueFromInputs<int64_t>(input_value_, kIndex2);
    if (!dim_opt.has_value()) {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", failed to get the input value of parameter 'dim'.";
      return FAILED;
    }
    if (dim_opt >= rank || dim_opt < -rank) {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", the value of parameter 'dim' is out of range ["
                    << (-rank) << ", " << (rank - 1) << "], the 'dim' is " << dim_opt.value() << " and the input.size()"
                    << " is " << rank << ".";
      return FAILED;
    }
    dim_ = dim_opt < 0 ? dim_opt.value() + rank : dim_opt.value();
  } else {
    // Set dim_ as -1 to represent the input 'dim' is None.
    dim_ = kInvalidDimValue;
  }
  return SUCCESS;
}

Status RepeatInterleaveInfo::CheckStrategy(const mindspore::parallel::StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", checkStrategy failed.";
    return FAILED;
  }
  auto strategies = strategy->GetInputDim();
  if (strategies.empty()) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the strategy is empty";
    return FAILED;
  }
  auto input_strategy = strategies[kIndex0];

  // when 'dim' is None, all dimensions cannot be split
  if (dim_ == kInvalidDimValue) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", all dimensions cannot be split when 'dim' is None.";
    return FAILED;
  }

  // when repeat.type is Tensor, the dimension 'dim' cannot be split
  if (is_tensor_repeat_) {
    if (input_strategy[dim_] != NO_SPLIT_STRATEGY) {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", the input's dimension 'dim' can not be split, the "
                    << "'dim' is " << dim_ << " and the shard strategy is " << input_strategy << ".";
      return FAILED;
    }
    auto repeat_strategy = strategies[kIndex1];
    if (repeat_strategy.size() != kSizeOne || repeat_strategy[kIndex0] != NO_SPLIT_STRATEGY) {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", the input 'repeat' cannot be split.";
      return FAILED;
    }
  }
  return SUCCESS;
}

void log_func_ele(const std::ostringstream &oss, bool is_in_layout_propagation) {
  if (is_in_layout_propagation) {
    MS_LOG(INFO) << oss.str();
  } else {
    MS_LOG(ERROR) << oss.str();
  }
}

Status RepeatInterleaveInfo::CheckInputLayout() {
  if (inputs_tensor_info_.size() != inputs_shape_.size()) {
    std::ostringstream oss;
    oss << "For distributed operator " << name_
        << ", the size of inputs_tensor_info should be equal to the num of tensor inputs, but the "
        << "inputs_tensor_info.size() is " << inputs_tensor_info_.size() << " and the num of tensor inputs is "
        << inputs_shape_.size() << ".";
    log_func_ele(oss, is_in_layout_propagation_);
  }

  auto input_tensor_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  dev_matrix_shape_ = input_tensor_layout.device_arrangement_origin().array();

  // when 'dim' is None, all dimensions cannot be split
  if (dim_ == kInvalidDimValue) {
    std::ostringstream oss;
    oss << "For distributed operator " << name_ << ", all dimensions cannot be split when 'dim' is None.";
    log_func_ele(oss, is_in_layout_propagation_);
    return FAILED;
  }

  // when repeat.type is Tensor, the dimension 'dim' cannot be split
  if (is_tensor_repeat_) {
    auto input_tensor_map = input_tensor_layout.tensor_map_before();
    if (input_tensor_map[dim_].empty()) {
      std::ostringstream oss;
      oss << "For distributed operator " << name_ << ", the layout of inputs' dimension 'dim' is empty.";
      log_func_ele(oss, is_in_layout_propagation_);
      return FAILED;
    }
    auto device_dims = SizeToLong(dev_matrix_shape_.size());
    auto shard_idx = device_dims - 1 - input_tensor_map[dim_][kIndex0];
    if (input_tensor_map[dim_].size() != kSizeOne || dev_matrix_shape_[shard_idx] != NO_SPLIT_STRATEGY) {
      std::ostringstream oss;
      oss << "For distributed operator " << name_ << ", the input's dimension 'dim' can not be split.";
      log_func_ele(oss, is_in_layout_propagation_);
      return FAILED;
    }
    auto repeat_tensor_layout = inputs_tensor_info_[kIndex1].tensor_layout();
    auto repeat_tensor_map = repeat_tensor_layout.tensor_map_before();
    if (repeat_tensor_map[kIndex0].empty()) {
      std::ostringstream oss;
      oss << "For distributed operator " << name_ << ", the layout of input 'repeats' is empty.";
      log_func_ele(oss, is_in_layout_propagation_);
      return FAILED;
    }
    shard_idx = device_dims - 1 - repeat_tensor_map[kIndex0][kIndex0];
    // the layout of 'repeats' such as ("a", "b") and (("a", "b")) is invalid
    if (repeat_tensor_map.size() != kSizeOne || repeat_tensor_map[kIndex0].size() != kSizeOne ||
        dev_matrix_shape_[shard_idx] != NO_SPLIT_STRATEGY) {
      std::ostringstream oss;
      oss << "For distributed operator " << name_ << ", the input 'repeat' cannot be split.";
      log_func_ele(oss, is_in_layout_propagation_);
      return FAILED;
    }
  }
  return SUCCESS;
}

Status RepeatInterleaveInfo::InferOutputTensorInfo() {
  auto input_tensor_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto input_tensor_map = input_tensor_layout.tensor_map_before();
  TensorLayout output_infer_tensor_layout;
  if ((output_infer_tensor_layout.InitFromExtendVector(dev_matrix_shape_, input_tensor_map, outputs_shape_[kIndex0]) !=
       SUCCESS)) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the output_tensor_layout init failed.";
    return FAILED;
  }
  if (output_infer_tensor_layout.tensor_shape_before().array() != outputs_shape_[kIndex0]) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the infer output shape "
                  << output_infer_tensor_layout.tensor_shape_before().array() << " does not match the output shape "
                  << outputs_shape_[kIndex0];
    return FAILED;
  }
  set_output_infer_tensor_layout(output_infer_tensor_layout);
  TensorInfo output_tensor_info(output_infer_tensor_layout);
  outputs_tensor_info_.push_back(output_tensor_info);
  return SUCCESS;
}

ReplaceGraphPtr RepeatInterleaveInfo::replace_graph(const CNodePtr &cnode) {
  if (inputs_tensor_info_[kIndex0].tensor_layout().IsInterleavedParallel()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << "For distributed operator " << name_ << " it does not support "
                                       << "interleaved parallel.";
  }
  return replace_graph_;
}

std::vector<StrategyPtr> RepeatInterleaveInfo::GenerateOpStrategies(int64_t stage_id) {
  Shapes splittable_inputs;
  if (dim_ == kInvalidDimValue) {
    Shape input0_split(inputs_shape_[kIndex0].size(), 0);
    splittable_inputs.push_back(input0_split);
  } else {
    int64_t rank = SizeToLong(inputs_shape_[kIndex0].size());
    if (dim_ < -rank || dim_ >= rank) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For distributed operator " << name_ << ", the value of parameter 'dim' is"
                                          << " out of range [" << (-rank) << ", " << (rank - 1) << "], the 'dim' is "
                                          << dim_ << " and the input.size() is " << rank << ".";
    }
    Shape input0_split(inputs_shape_[0].size(), 1);
    splittable_inputs.push_back(input0_split);
    if (is_tensor_repeat_) {
      splittable_inputs[kIndex0][dim_] = 0;
      splittable_inputs.push_back({0});
    }
  }

  std::vector<StrategyPtr> sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape_, splittable_inputs, &sp_vector) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For distributed operator " << name_ << ", generate strategies failed.";
  }
  if (sp_vector.empty()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For distributed operator " << name_ << ", No available strategy";
  }

  return sp_vector;
}

Status RepeatInterleaveTensorInfo::InferTensorMap() {
  if (ActivationBase::InferTensorMap() != SUCCESS) {
    return FAILED;
  }
  inputs_tensor_map_.push_back({MAP_NONE});  // repeats
  return SUCCESS;
}

REGISTER(ExpInfo);
REGISTER(LogInfo);
REGISTER(CosInfo);
REGISTER(ACosInfo);
REGISTER(LogicalNotInfo);
REGISTER(AbsInfo);
REGISTER(SignInfo);
REGISTER(FloorInfo);
REGISTER(RoundInfo);
REGISTER(ReciprocalInfo);
REGISTER(InvInfo);
REGISTER(RsqrtInfo);
REGISTER(TanInfo);
REGISTER(SinInfo);
REGISTER(SinhInfo);
REGISTER(Log1pInfo);
REGISTER(Expm1Info);
REGISTER(CoshInfo);
REGISTER(CeilInfo);
REGISTER(CholeskyInfo);
REGISTER(AtanhInfo);
REGISTER(AtanInfo);
REGISTER(AsinInfo);
REGISTER(AsinhInfo);
REGISTER(AcoshInfo);
REGISTER(ErfInfo);
REGISTER(ErfcInfo);
REGISTER(ZerosLikeInfo);
REGISTER(OnesLikeInfo);
REGISTER(BesselI0eInfo);
REGISTER(BesselI1eInfo);
REGISTER(BesselI0Info);
REGISTER(BesselI1Info);
REGISTER(BesselJ0Info);
REGISTER(BesselJ1Info);
REGISTER(LgammaInfo);
REGISTER(TruncInfo);
REGISTER(RepeatInterleaveInfo);
REGISTER(RepeatInterleaveIntInfo);
REGISTER(RepeatInterleaveTensorInfo);
}  // namespace parallel
}  // namespace mindspore
