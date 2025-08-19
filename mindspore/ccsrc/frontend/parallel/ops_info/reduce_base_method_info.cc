/**
 * Copyright 2023-2025 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/reduce_base_method_info.h"

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "frontend/parallel/device_manager.h"
#include "frontend/parallel/device_matrix.h"
#include "frontend/parallel/dynamic_creator.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"
#include "ir/value.h"
#include "ops_utils/op_utils.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr auto kNameAxis = "axis";
constexpr auto kNameKeepDims = "keep_dims";
constexpr auto kNameDim = "dim";
constexpr auto kNameKeepDim = "keepdim";

bool IsDataParallelStrategy(const Dimensions &strategy, int32_t stage_id) {
  CheckGlobalDeviceManager();
  size_t total_dev_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
  if (strategy.empty()) {
    MS_LOG(EXCEPTION) << "IsDataParallelStrategy: strategy is empty";
  }

  return (LongToSize(strategy[0]) == total_dev_num);
}
}  // namespace

Status ReduceBaseMethod::InferMirrorOps() {
  mirror_ops_.clear();
  Shape input_tensor_map = inputs_tensor_map_.at(0);
  std::vector<Group> input_group;
  if (CreateGroupByTensorMap(input_tensor_map, &input_group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }

  if (input_group.empty()) {
    MS_LOG(INFO) << name_ << ": The mirror ops is empty.";
    return SUCCESS;
  } else {
    auto op_for_weight = CreateMirrorOps(input_group[0].name(), input_group[0].GetDevNum());
    mirror_ops_.push_back(op_for_weight);

    OperatorVector op_helper;
    auto prim_name = GetPrimNameFromInfoName(name_);
    auto res_size = ops::GetOpInputsNum(prim_name) - mirror_ops_.size();
    for (size_t i = 0; i < res_size; ++i) {
      mirror_ops_.push_back(op_helper);
    }

    std::string group_name = input_group[0].name();
    MS_LOG(INFO) << name_ << ": Create the mirror ops for weight success, the group is " << group_name;
  }

  return SUCCESS;
}

std::vector<int64_t> ReduceBaseMethod::reduce_dim() {
  std::vector<int64_t> dim_list;
  std::vector<int64_t> axis_value;
  auto axis_opt = GetArrayValueFromInputs<int64_t>(input_value_, name_, kNameAxis);
  if (!axis_opt.has_value()) {
    if (!outputs_shape_.empty() && outputs_shape_[0].empty()) {
      // the axis of reduce op may be None, it is equivalent to axis = ().
      axis_value = {};
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For " << name_ << ", failed to get value for " << kNameAxis << ".";
    }
  } else {
    axis_value = axis_opt.value();
  }

  MS_ASSERT(inputs_shape_.size() >= 1);
  auto x_dim = inputs_shape_.at(0).size();
  // axis is (), reduce all dim
  if (axis_value.empty()) {
    for (size_t i = 0; i < x_dim; ++i) {
      dim_list.push_back(SizeToLong(i));
    }
  } else {
    auto AxisCorrectFunc = [x_dim](const int64_t axis) {
      if (axis < 0) {
        return axis + SizeToLong(x_dim);
      }
      return axis;
    };
    std::transform(axis_value.begin(), axis_value.end(), std::back_inserter(dim_list), AxisCorrectFunc);
  }
  return dim_list;
}

Status ReduceBaseMethod::GetAttrs() {
  // get attr cross_batch and keep_dims
  auto keep_dims_opt = GetScalarValueFromInputs<bool>(input_value_, name_, kNameKeepDims);
  if (!keep_dims_opt.has_value()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For " << name_ << ", failed to get value for " << kNameKeepDims << ".";
  }
  keepdims_ = keep_dims_opt.value();

  auto cross_batch_iter = attrs_.find(CROSS_BATCH);
  if (cross_batch_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(cross_batch_iter->second);
    if (!cross_batch_iter->second->isa<BoolImm>()) {
      MS_LOG(ERROR) << name_ << ": cross_batch is not a bool.";
      return FAILED;
    }
    cross_batch_ = cross_batch_iter->second->cast<BoolImmPtr>()->value();
  }
  auto reducemethodcost = std::dynamic_pointer_cast<ReduceMethodCost>(operator_cost());
  if (reducemethodcost == nullptr) {
    MS_LOG(ERROR) << "Cost cast to ReduceMethodCostPtr failed!";
    return FAILED;
  }
  reducemethodcost->set_cross_batch(cross_batch_);
  return SUCCESS;
}

Status ReduceMeanInfo::InferForwardCommunication() {
  auto strategies = strategy_->GetInputDim();
  Dimensions stra = strategies.at(0);
  if (cross_batch_ && IsDataParallelStrategy(stra, stage_id_)) {
    MS_LOG(INFO) << name_ << ": cross_batch is True, don't need to InferForwardCommunication";
    return SUCCESS;
  }
  forward_op_.clear();
  std::vector<int64_t> reduce_dims = reduce_dim();
  auto input_tensor_map = inputs_tensor_map_.at(kIndex0);
  std::vector<int64_t> shard_dims;
  for (size_t i = 0; i < input_tensor_map.size(); ++i) {
    if (std::find_if(reduce_dims.begin(), reduce_dims.end(), [i](const int64_t dim) { return SizeToLong(i) == dim; }) !=
        reduce_dims.end()) {
      shard_dims.push_back(SizeToLong(dev_matrix_shape_.size() - kIndex1 - input_tensor_map[i]));
    }
  }

  RankList comm_rank_list;
  auto device_matrix =
    DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(), dev_matrix_shape_);
  if (device_matrix.GetDevicesAlongMultiDim(shard_dims, &comm_rank_list) != SUCCESS) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", infer Forward communication by multi dim failed.";
    return FAILED;
  }
  if (comm_rank_list.size() <= 1) {
    MS_LOG(INFO) << "For distributed operator " << name_ << ", forward communication is not required.";
    return SUCCESS;
  }
  Group comm_group;
  if (g_device_manager->CreateGroup(comm_rank_list, &comm_group) != SUCCESS) {
    MS_LOG(ERROR) << "For distributed operator " << name_
                  << ", create communication group by comm_rank_list failed, the communication rank_list is: "
                  << comm_rank_list << ", the full_name of node is: " << cnode_->fullname_with_scope();
    return FAILED;
  }

  auto tensor_type_ptr = outputs_dtype_->cast<mindspore::TensorTypePtr>();
  if (tensor_type_ptr == nullptr) {
    MS_LOG(ERROR) << name_ << ": Failed to cast outputs_dtype_ to TensorTypePtr. The pointer is null.";
    return FAILED;
  }
  auto element_type = tensor_type_ptr->element();
  forward_op_ = CreateAllReduceMeanForwardOp(comm_group, element_type);
  MS_LOG(INFO) << "For distributed operator " << name_ << ", the group name of forward communication is "
               << comm_group.name() << ".";
  return SUCCESS;
}

std::vector<int64_t> MeanExtInfo::reduce_dim() {
  std::vector<int64_t> dim_list{};
  auto prim_name = GetPrimNameFromInfoName(name_);
  auto idx = ops::GetInputIndexByName(prim_name, kNameDim);
  if (input_value_.size() <= idx || input_value_[idx] == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For " << name_ << ", the input_value_ is less than " << idx
                                        << ", or input_value_[idx] == nullptr.";
  }
  std::vector<int64_t> axis_value;
  if (input_value_[idx]->isa<None>()) {
    axis_value = {};
  } else {
    auto axis_opt = GetArrayValueFromInputs<int64_t>(input_value_, name_, kNameDim);
    if (!axis_opt.has_value()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For " << name_ << ", failed to get value for " << kNameDim << ".";
    }
    axis_value = axis_opt.value();
  }
  MS_ASSERT(inputs_shape_.size() >= 1);
  auto x_dim = inputs_shape_.at(0).size();
  // axis is (), reduce all dim
  if (axis_value.empty()) {
    for (size_t i = 0; i < x_dim; ++i) {
      dim_list.push_back(SizeToLong(i));
    }
  } else {
    auto AxisCorrectFunc = [x_dim](const int64_t axis) {
      if (axis < 0) {
        return axis + SizeToLong(x_dim);
      }
      return axis;
    };
    std::transform(axis_value.begin(), axis_value.end(), std::back_inserter(dim_list), AxisCorrectFunc);
  }
  return dim_list;
}

Status MeanExtInfo::GetAttrs() {
  // get attr cross_batch and keep_dims
  auto keep_dims_opt = GetScalarValueFromInputs<bool>(input_value_, name_, kNameKeepDim);
  if (!keep_dims_opt.has_value()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For " << name_ << ", failed to get value for " << kNameKeepDim << ".";
  }
  keepdims_ = keep_dims_opt.value();

  auto cross_batch_iter = attrs_.find(CROSS_BATCH);
  if (cross_batch_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(cross_batch_iter->second);
    if (!cross_batch_iter->second->isa<BoolImm>()) {
      MS_LOG(ERROR) << name_ << ": cross_batch is not a bool.";
      return FAILED;
    }
    cross_batch_ = cross_batch_iter->second->cast<BoolImmPtr>()->value();
  }
  auto reducemethodcost = std::dynamic_pointer_cast<ReduceMethodCost>(operator_cost());
  if (reducemethodcost == nullptr) {
    MS_LOG(ERROR) << "Cost cast to ReduceMethodCostPtr failed!";
    return FAILED;
  }
  reducemethodcost->set_cross_batch(cross_batch_);
  return SUCCESS;
}

std::vector<StrategyPtr> MeanExtInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
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

Status MeanExtInfo::CheckInputLayout() {
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
  return SUCCESS;
}

Status MeanExtInfo::InferOutputTensorInfo() {
  auto input_tensor_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto input_tensor_map = input_tensor_layout.tensor_map_before();
  dev_matrix_shape_ = input_tensor_layout.device_arrangement_origin().array();
  size_t size = input_tensor_map.size();

  std::vector<int64_t> dim_list = reduce_dim();
  Shapes outputs_tensor_map;
  for (size_t i = 0; i < size; ++i) {
    if (std::find(dim_list.begin(), dim_list.end(), SizeToLong(i)) != dim_list.end()) {
      if (keepdims_) {
        outputs_tensor_map.push_back({-1});
      } else {
        continue;
      }
    } else {
      outputs_tensor_map.push_back(input_tensor_map[i]);
    }
  }

  TensorLayout output_infer_tensor_layout;
  if ((output_infer_tensor_layout.InitFromExtendVector(dev_matrix_shape_, outputs_tensor_map,
                                                       outputs_shape_[kIndex0]) != SUCCESS)) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the output_tensor_layout init failed.";
    return FAILED;
  }
  if (output_infer_tensor_layout.tensor_shape_before().array() != outputs_shape_[kIndex0]) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the infer output shape "
                  << output_infer_tensor_layout.tensor_shape_before().array() << " does not match the output shape "
                  << outputs_shape_[kIndex0];
    return FAILED;
  }
  TensorInfo output_tensor_info(output_infer_tensor_layout);
  outputs_tensor_info_.push_back(output_tensor_info);
  is_infer_out_layout_ = true;
  return SUCCESS;
}

Status MeanExtInfo::CheckOutputLayout() {
  if (outputs_tensor_info_.size() != kSizeOne) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_ << ", the size of output_tensor_layout for " << name_
                   << " is " << outputs_tensor_info_.size() << " rather than 1.";
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", the size of output_tensor_layout for " << name_
                    << " is " << outputs_tensor_info_.size() << " rather than 1.";
    }
    return FAILED;
  }
  if (!is_infer_out_layout_) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_ << ", parameter of output tensor layout for " << name_
                   << " is not allowed to be set by users.";
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", parameter of output tensor layout for " << name_
                    << " is not allowed to be set by users.";
    }
    return FAILED;
  }
  MS_LOG(DEBUG) << "For distributed operator " << name_ << ", using output tensor layout infer by input tensor layout.";
  return SUCCESS;
}

Status MeanExtInfo::InferForwardCommunicationByLayout() {
  forward_op_.clear();
  auto input_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto input_tensor_map = input_layout.tensor_map_before();

  std::vector<int64_t> dim_list = reduce_dim();
  std::vector<int64_t> shard_dims;
  for (size_t i = 0; i < input_tensor_map.size(); ++i) {
    auto pos = std::find_if(dim_list.begin(), dim_list.end(), [i](const int64_t &dim) { return SizeToLong(i) == dim; });
    if (pos != dim_list.end()) {
      std::for_each(input_tensor_map[i].begin(), input_tensor_map[i].end(), [this, &shard_dims](auto elem) {
        if (elem != -1) {
          shard_dims.push_back(SizeToLong(dev_matrix_shape_.size() - kIndex1 - elem));
        }
      });
    }
  }

  RankList comm_rank_list;
  auto device_matrix =
    DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(), dev_matrix_shape_);
  if (device_matrix.GetDevicesAlongMultiDim(shard_dims, &comm_rank_list) != SUCCESS) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", infer Forward communication by multi dim failed.";
    return FAILED;
  }
  if (comm_rank_list.size() == 1) {
    MS_LOG(INFO) << "For distributed operator " << name_ << ", forward communication is not required.";
    return SUCCESS;
  }
  Group comm_group;
  if (g_device_manager->CreateGroup(comm_rank_list, &comm_group) != SUCCESS) {
    MS_LOG(ERROR) << "For distributed operator " << name_
                  << ", create communication group by comm_rank_list failed, the communication rank_list is: "
                  << comm_rank_list << ", the full_name of node is: " << cnode_->fullname_with_scope();
    return FAILED;
  }

  auto tensor_type_ptr = outputs_dtype_->cast<mindspore::TensorTypePtr>();
  if (tensor_type_ptr == nullptr) {
    MS_LOG(ERROR) << name_ << ": Failed to cast outputs_dtype_ to TensorTypePtr. The pointer is null.";
    return FAILED;
  }
  auto element_type = tensor_type_ptr->element();
  forward_op_ = CreateAllReduceMeanForwardOp(comm_group, element_type);
  MS_LOG(INFO) << "For distributed operator " << name_ << ", the group name of forward communication is "
               << comm_group.name() << ".";
  return SUCCESS;
}

ForwardOp ReduceAnyInfo::CreateForwardOp(const std::vector<Group> &forward_group) const {
  // Create Cast to Int32 op
  Operator op0 = CreateCastOp(kInt32);

  // Create AllReduce op
  Operator op1 = CreateAllReduceOp(reduce_method_, forward_group[0].name());
  std::string group_name = forward_group[0].name();
  MS_LOG(INFO) << "The group of forward all reduce is " << group_name << ", method is " << reduce_method_;

  // Create Cast to Bool op
  Operator op2 = CreateCastOp(kBool);

  ForwardOp forward_op = {op0, op1, op2};

  return forward_op;
}

Status ReduceAnyInfo::InferForwardCommunication() {
  auto strategies = strategy_->GetInputDim();
  Dimensions stra = strategies.at(0);
  if (cross_batch_ && IsDataParallelStrategy(stra, stage_id_)) {
    MS_LOG(INFO) << name_ << ": cross_batch is True, don't need to InferForwardCommunication";
    return SUCCESS;
  }
  forward_op_.clear();
  std::vector<int64_t> dim_list = reduce_dim();
  size_t size = stra.size();
  // judge if the reduce dim is partitioned.
  Shape group_creat_map;

  // if repeated calculation and the repeated_calc_num_ insert to the first dimension of dev matrix,
  // it need to handle the first dimension of map.
  if ((dev_matrix_shape_.size() > size) && !repeated_num_in_dev_matrix_right_) {
    group_creat_map.push_back(SizeToInt(dev_matrix_shape_.size() - size_t(1)));
  }

  for (size_t index = 0; index < size; ++index) {
    auto pos =
      std::find_if(dim_list.begin(), dim_list.end(), [index](const int64_t &dim) { return SizeToLong(index) == dim; });
    if (pos != dim_list.end() && stra[index] != 1) {
      continue;
    }
    group_creat_map.push_back(SizeToLong(size) - SizeToLong(index) - 1);
  }

  // if repeated calculation and the repeated_calc_num_ insert to the last dimension of dev matrix,
  // it need to handle the group_creat_map and insert the 0 to the last dimension of the group_creat_map.
  if (repeated_num_in_dev_matrix_right_ && (repeated_calc_num_ > 1)) {
    for (auto &ele : group_creat_map) {
      if (ele == MAP_NONE) {
        continue;
      }
      ele += 1;
    }
    group_creat_map.push_back(0);
  }

  std::vector<Group> forward_group;
  if (CreateGroupByTensorMap(group_creat_map, &forward_group) != SUCCESS) {
    ReportError(name_ + ": Create group failed.");
    return FAILED;
  }
  if (!forward_group.empty()) {
    forward_op_ = CreateForwardOp(forward_group);
  }

  return SUCCESS;
}

std::vector<int64_t> SumExtInfo::reduce_dim() {
  std::vector<int64_t> dim_list{};
  auto prim_name = GetPrimNameFromInfoName(name_);
  auto idx = ops::GetInputIndexByName(prim_name, kNameDim);
  if (input_value_.size() <= idx || input_value_[idx] == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For " << name_ << ", the input_value_ is less than " << idx
                                        << ", or input_value_[idx] == nullptr.";
  }
  std::vector<int64_t> axis_value;
  if (input_value_[idx]->isa<None>()) {
    axis_value = {};
  } else {
    auto axis_opt = GetArrayValueFromInputs<int64_t>(input_value_, name_, kNameDim);
    if (!axis_opt.has_value()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For " << name_ << ", failed to get value for " << kNameDim << ".";
    }
    axis_value = axis_opt.value();
  }
  MS_ASSERT(inputs_shape_.size() >= 1);
  auto x_dim = inputs_shape_.at(0).size();
  // axis is (), reduce all dim
  if (axis_value.empty()) {
    for (size_t i = 0; i < x_dim; ++i) {
      dim_list.push_back(SizeToLong(i));
    }
  } else {
    auto AxisCorrectFunc = [x_dim](const int64_t axis) {
      if (axis < 0) {
        return axis + SizeToLong(x_dim);
      }
      return axis;
    };
    std::transform(axis_value.begin(), axis_value.end(), std::back_inserter(dim_list), AxisCorrectFunc);
  }
  return dim_list;
}

Status SumExtInfo::GetAttrs() {
  // get attr cross_batch and keep_dims
  auto keep_dims_opt = GetScalarValueFromInputs<bool>(input_value_, name_, kNameKeepDim);
  if (!keep_dims_opt.has_value()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "For " << name_ << ", failed to get value for " << kNameKeepDim << ".";
  }
  keepdims_ = keep_dims_opt.value();

  auto cross_batch_iter = attrs_.find(CROSS_BATCH);
  if (cross_batch_iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(cross_batch_iter->second);
    if (!cross_batch_iter->second->isa<BoolImm>()) {
      MS_LOG(ERROR) << name_ << ": cross_batch is not a bool.";
      return FAILED;
    }
    auto cross_batch = cross_batch_iter->second->cast<BoolImmPtr>();
    MS_EXCEPTION_IF_NULL(cross_batch);
    cross_batch_ = cross_batch->value();
  }
  auto reducemethodcost = std::dynamic_pointer_cast<ReduceMethodCost>(operator_cost());
  if (reducemethodcost == nullptr) {
    MS_LOG(ERROR) << "Cost cast to ReduceMethodCostPtr failed!";
    return FAILED;
  }
  reducemethodcost->set_cross_batch(cross_batch_);
  return SUCCESS;
}

std::vector<StrategyPtr> SumExtInfo::GenerateOpStrategies(int64_t stage_id) {
  Shape input0_split(inputs_shape_[0].size(), 1);
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

Status SumExtInfo::CheckInputLayout() {
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
  return SUCCESS;
}

Status SumExtInfo::InferOutputTensorInfo() {
  auto input_tensor_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto input_tensor_map = input_tensor_layout.tensor_map_before();
  dev_matrix_shape_ = input_tensor_layout.device_arrangement_origin().array();
  size_t size = input_tensor_map.size();

  std::vector<int64_t> dim_list = reduce_dim();
  Shapes outputs_tensor_map;
  for (size_t i = 0; i < size; ++i) {
    if (std::find(dim_list.begin(), dim_list.end(), SizeToLong(i)) != dim_list.end()) {
      if (keepdims_) {
        outputs_tensor_map.push_back({-1});
      } else {
        continue;
      }
    } else {
      outputs_tensor_map.push_back(input_tensor_map[i]);
    }
  }

  TensorLayout output_infer_tensor_layout;
  if ((output_infer_tensor_layout.InitFromExtendVector(dev_matrix_shape_, outputs_tensor_map,
                                                       outputs_shape_[kIndex0]) != SUCCESS)) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the output_tensor_layout init failed.";
    return FAILED;
  }
  if (output_infer_tensor_layout.tensor_shape_before().array() != outputs_shape_[kIndex0]) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", the infer output shape "
                  << output_infer_tensor_layout.tensor_shape_before().array() << " does not match the output shape "
                  << outputs_shape_[kIndex0];
    return FAILED;
  }
  TensorInfo output_tensor_info(output_infer_tensor_layout);
  outputs_tensor_info_.push_back(output_tensor_info);
  is_infer_out_layout_ = true;
  return SUCCESS;
}

Status SumExtInfo::CheckOutputLayout() {
  if (outputs_tensor_info_.size() != kSizeOne) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_ << ", the size of output_tensor_layout for " << name_
                   << " is " << outputs_tensor_info_.size() << " rather than 1.";
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", the size of output_tensor_layout for " << name_
                    << " is " << outputs_tensor_info_.size() << " rather than 1.";
    }
    return FAILED;
  }
  if (!is_infer_out_layout_) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_ << ", parameter of output tensor layout for " << name_
                   << " is not allowed to be set by users.";
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", parameter of output tensor layout for " << name_
                    << " is not allowed to be set by users.";
    }
    return FAILED;
  }
  MS_LOG(DEBUG) << "For distributed operator " << name_ << ", using output tensor layout infer by input tensor layout.";
  return SUCCESS;
}

Status SumExtInfo::InferForwardCommunicationByLayout() {
  forward_op_.clear();
  auto input_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto input_tensor_map = input_layout.tensor_map_before();

  std::vector<int64_t> dim_list = reduce_dim();
  std::vector<int64_t> shard_dims;
  for (size_t i = 0; i < input_tensor_map.size(); ++i) {
    // use to generate group_rank_id
    auto pos = std::find_if(dim_list.begin(), dim_list.end(), [i](const int64_t &dim) { return SizeToLong(i) == dim; });
    if (pos != dim_list.end()) {
      std::transform(input_tensor_map[i].begin(), input_tensor_map[i].end(), std::back_inserter(shard_dims),
                     [this](auto elem) { return SizeToLong(dev_matrix_shape_.size() - kIndex1 - elem); });
    }
  }

  RankList comm_rank_list;
  auto device_matrix =
    DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(), dev_matrix_shape_);
  if (device_matrix.GetDevicesAlongMultiDim(shard_dims, &comm_rank_list) != SUCCESS) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", infer Forward communication by multi dim failed.";
    return FAILED;
  }
  if (comm_rank_list.size() == 1) {
    MS_LOG(INFO) << "For distributed operator " << name_ << ", forward communication is not required.";
    return SUCCESS;
  }
  Group comm_group;
  if (g_device_manager->CreateGroup(comm_rank_list, &comm_group) != SUCCESS) {
    MS_LOG(ERROR) << "For distributed operator " << name_
                  << ", create communication group by comm_rank_list failed, the communication rank_list is: "
                  << comm_rank_list << ", the full_name of node is: " << cnode_->fullname_with_scope();
    return FAILED;
  }

  Operator op = CreateAllReduceOp(reduce_method_, comm_group.name());
  forward_op_.push_back(op);
  MS_LOG(INFO) << "For distributed operator " << name_ << ", the group name of forward communication is "
               << comm_group.name() << ".";
  return SUCCESS;
}

std::vector<int64_t> MaxInfo::reduce_dim() {
  std::vector<int64_t> dim_list;
  MS_ASSERT(inputs_shape_.size() == 1);
  auto input_dim = inputs_shape_.at(0).size();
  // max ops does not have input 'dim', reduce all dim
  for (size_t i = 0; i < input_dim; ++i) {
    dim_list.push_back(SizeToLong(i));
  }
  return dim_list;
}

Status MaxInfo::InferForwardCommunicationByLayout() {
  forward_op_.clear();
  auto input_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  auto input_tensor_map = input_layout.origin_tensor_map().array();
  std::vector<int64_t> shard_dims;
  for (size_t i = 0; i < input_tensor_map.size(); ++i) {
    shard_dims.push_back(SizeToLong(dev_matrix_shape_.size() - kIndex1 - input_tensor_map[i]));
  }
  RankList comm_rank_list;
  auto device_matrix =
    DeviceMatrix(g_device_manager->global_rank(), g_device_manager->GetDeviceListInThisStage(), dev_matrix_shape_);
  if (device_matrix.GetDevicesAlongMultiDim(shard_dims, &comm_rank_list) != SUCCESS) {
    MS_LOG(ERROR) << "For distributed operator " << name_ << ", infer Forward communication by multi dim failed.";
    return FAILED;
  }
  if (comm_rank_list.size() == 1) {
    MS_LOG(INFO) << "For distributed operator " << name_ << ", forward communication is not required.";
    return SUCCESS;
  }
  Group comm_group;
  if (g_device_manager->CreateGroup(comm_rank_list, &comm_group) != SUCCESS) {
    MS_LOG(ERROR) << "For distributed operator " << name_
                  << ", create communication group by comm_rank_list failed, the communication rank_list is: "
                  << comm_rank_list << ", the full_name of node is: " << cnode_->fullname_with_scope();
    return FAILED;
  }

  Operator op = CreateAllReduceOp(reduce_method_, comm_group.name());
  forward_op_.push_back(op);
  MS_LOG(INFO) << "For distributed operator " << name_ << ", the group name of forward communication is "
               << comm_group.name() << ".";
  return SUCCESS;
}

Status MaxInfo::CheckInputLayout() {
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
  return SUCCESS;
}

Status MaxInfo::InferOutputTensorInfo() {
  auto input_tensor_layout = inputs_tensor_info_[kIndex0].tensor_layout();
  dev_matrix_shape_ = input_tensor_layout.device_arrangement_origin().array();
  // Max ops reduce all the dims and output is a single num, so the output tensor map is empty;
  Shape outputs_tensor_map = {};
  TensorLayout output_infer_tensor_layout;
  if ((output_infer_tensor_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map, outputs_shape_[kIndex0]) !=
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
  TensorInfo output_tensor_info(output_infer_tensor_layout);
  outputs_tensor_info_.push_back(output_tensor_info);
  is_infer_out_layout_ = true;
  return SUCCESS;
}

Status MaxInfo::CheckOutputLayout() {
  if (outputs_tensor_info_.size() != kSizeOne) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_ << ", the size of output_tensor_layout for " << name_
                   << " is " << outputs_tensor_info_.size() << " rather than 1.";
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", the size of output_tensor_layout for " << name_
                    << " is " << outputs_tensor_info_.size() << " rather than 1.";
    }
    return FAILED;
  }
  if (!is_infer_out_layout_) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "For distributed operator " << name_ << ", parameter of output tensor layout for " << name_
                   << " is not allowed to be set by users.";
    } else {
      MS_LOG(ERROR) << "For distributed operator " << name_ << ", parameter of output tensor layout for " << name_
                    << " is not allowed to be set by users.";
    }
    return FAILED;
  }
  MS_LOG(DEBUG) << "For distributed operator " << name_ << ", using output tensor layout infer by input tensor layout.";
  return SUCCESS;
}

REGISTER(ReduceMaxInfo);
REGISTER(ReduceMeanInfo);
REGISTER(ReduceSumInfo);
REGISTER(ReduceAnyInfo);
REGISTER(ReduceMinInfo);
REGISTER(ReduceProdInfo);
REGISTER(ReduceAllInfo);
REGISTER(SumExtInfo);
REGISTER(MaxInfo);
REGISTER(MeanExtInfo);
}  // namespace parallel
}  // namespace mindspore
