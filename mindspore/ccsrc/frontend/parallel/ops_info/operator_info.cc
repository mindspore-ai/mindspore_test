/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/ops_info/operator_info.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <set>

#include "frontend/parallel/auto_parallel/edge_costmodel.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "include/common/debug/anf_dump_utils.h"
#include "include/common/utils/parallel_context.h"
#include "ir/tensor.h"
#include "ir/value.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "op_def/arithmetic_ops.h"

namespace mindspore {
namespace parallel {
namespace {
struct InStrategyValueRegister {
  InStrategyValueRegister() noexcept {
    AnfDumpHandler::SetInStrategyValueHandler([](const std::shared_ptr<AnfNode> &node) -> ValuePtr {
      auto operator_info = node->user_data<parallel::OperatorInfo>();
      if (operator_info == nullptr) {
        return nullptr;
      }

      auto in_strategy = operator_info->strategy();
      if (in_strategy == nullptr) {
        return nullptr;
      }

      return MakeValue(in_strategy->GetInputDim());
    });
  }
} in_regist;

struct InStrategyStageValueRegister {
  InStrategyStageValueRegister() noexcept {
    AnfDumpHandler::SetInStrategyStageValueHandler([](const std::shared_ptr<AnfNode> &node) -> ValuePtr {
      auto operator_info = node->user_data<parallel::OperatorInfo>();
      if (operator_info == nullptr) {
        return nullptr;
      }

      auto in_strategy = operator_info->strategy();
      if (in_strategy == nullptr) {
        return nullptr;
      }

      return MakeValue(in_strategy->GetInputStage());
    });
  }
} in_stage_regist;

struct OutStrategyValueRegister {
  OutStrategyValueRegister() noexcept {
    AnfDumpHandler::SetOutStrategyValueHandler([](const std::shared_ptr<AnfNode> &node) -> ValuePtr {
      auto operator_info = node->user_data<parallel::OperatorInfo>();
      if (operator_info == nullptr) {
        return nullptr;
      }

      auto in_strategy = operator_info->out_strategy();
      if (in_strategy == nullptr) {
        return nullptr;
      }

      return MakeValue(in_strategy->GetInputDim());
    });
  }
} out_regist;

struct InLayoutValueRegister {
  InLayoutValueRegister() noexcept {
    AnfDumpHandler::SetInLayoutValueHandler([](const std::shared_ptr<AnfNode> &node) -> ValueTuplePtr {
      auto operator_info = node->user_data<parallel::OperatorInfo>();
      if (operator_info == nullptr) {
        return nullptr;
      }

      auto tensor_infos = operator_info->inputs_tensor_info();
      if (tensor_infos.empty()) {
        return nullptr;
      }

      std::vector<ValuePtr> result;
      for (const TensorInfo &tensor_info : tensor_infos) {
        std::vector<std::pair<ValuePtr, ValuePtr>> key_values;
        auto tensor_layout = tensor_info.tensor_layout();

        Arrangement device_mat = tensor_layout.device_arrangement_origin();
        ValuePtr device_matrix_key = MakeValue<std::string>(DEVICE_MATRIX);
        ValuePtr device_matrix_value = MakeValue<Shape>(device_mat.array());
        key_values.emplace_back(device_matrix_key, device_matrix_value);

        Map tensor_map = tensor_layout.origin_tensor_map();
        ValuePtr tensor_map_key = MakeValue<std::string>(TENSOR_MAP);
        ValuePtr tensor_map_value = MakeValue<Shape>(tensor_map.array());
        key_values.emplace_back(tensor_map_key, tensor_map_value);

        result.push_back(std::make_shared<ValueDictionary>(key_values));
      }
      return std::make_shared<ValueTuple>(result);
    });
  }
} in_layout_regist;

struct OutLayoutValueRegister {
  OutLayoutValueRegister() noexcept {
    AnfDumpHandler::SetOutLayoutValueHandler([](const std::shared_ptr<AnfNode> &node) -> ValueTuplePtr {
      auto operator_info = node->user_data<parallel::OperatorInfo>();
      if (operator_info == nullptr) {
        return nullptr;
      }

      auto tensor_infos = operator_info->outputs_tensor_info();
      if (tensor_infos.empty()) {
        return nullptr;
      }

      std::vector<ValuePtr> result;
      for (const TensorInfo &tensor_info : tensor_infos) {
        std::vector<std::pair<ValuePtr, ValuePtr>> key_values;
        auto tensor_layout = tensor_info.tensor_layout();

        Arrangement device_mat = tensor_layout.device_arrangement_origin();
        ValuePtr device_matrix_key = MakeValue<std::string>(DEVICE_MATRIX);
        ValuePtr device_matrix_value = MakeValue<Shape>(device_mat.array());
        key_values.emplace_back(device_matrix_key, device_matrix_value);

        Map tensor_map = tensor_layout.origin_tensor_map();
        ValuePtr tensor_map_key = MakeValue<std::string>(TENSOR_MAP);
        ValuePtr tensor_map_value = MakeValue<Shape>(tensor_map.array());
        key_values.emplace_back(tensor_map_key, tensor_map_value);

        result.push_back(std::make_shared<ValueDictionary>(key_values));
      }
      return std::make_shared<ValueTuple>(result);
    });
  }
} out_layout_regist;
}  // namespace

std::string StrategyToString(const Strategies &strategy) {
  std::string strategy_str = "";
  strategy_str += "(";
  for (size_t i = 0; i < strategy.size(); ++i) {
    strategy_str += "(";
    for (size_t j = 0; j < strategy[i].size(); ++j) {
      strategy_str += std::to_string(strategy[i][j]);
      if (j != strategy[i].size() - 1) {
        strategy_str += ", ";
      }
    }
    strategy_str += ")";
    if (i != strategy.size() - 1) {
      strategy_str += ", ";
    }
  }
  if (strategy.size() == 1) {
    strategy_str += ",";
  }
  strategy_str += ")";
  return strategy_str;
}

Status OperatorInfo::CheckOutputStrategy(const StrategyPtr &out_strategy) {
  if (out_strategy && name_.find("ShardIdentity") == std::string::npos && !AttrFound(attrs_, CELL_SHARD_OP)) {
    MS_LOG(ERROR) << name_ << ": It does not support to set output strategy now, please modify the shard set";
    return FAILED;
  }
  return SUCCESS;
}

Status OperatorInfo::CheckStrategyByVector(const Shapes &stra, const Shapes &inputs_shape) {
  size_t strategy_size = stra.size();
  size_t inputs_shape_size = inputs_shape.size();
  if (strategy_size != inputs_shape_size) {
    MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(stra) << ", strategy size: " << strategy_size
                  << " is not equal to inputs size: " << inputs_shape_size;
    return FAILED;
  }

  for (size_t i = 0; i < strategy_size; ++i) {
    Shape sub_strategy = stra.at(i);
    Shape sub_input_shape = inputs_shape.at(i);
    size_t strategy_len = sub_strategy.size();
    size_t inputs_len = sub_input_shape.size();
    MS_LOG(DEBUG) << "Compare: sub_input_shape:" << sub_input_shape << " sub_strategy: " << sub_strategy;
    if (strategy_len != inputs_len) {
      MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(stra) << ", strategy len: " << strategy_len
                    << " is not equal to inputs len: " << inputs_len << ", index: " << i;
      return FAILED;
    }

    for (size_t j = 0; j < strategy_len; ++j) {
      int64_t strategy_value = sub_strategy.at(j);
      if (strategy_value < MIN_SLICE_NUM) {
        MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(stra)
                      << ", the value of strategy must be larger than 0, but get " << strategy_value;
        return FAILED;
      }

      int64_t shape_value = sub_input_shape.at(j);
      if (shape_value != -1 && (shape_value % strategy_value) != 0) {
        if (dynamic_shape_flag_) {
          Shapes origin_shapes = inputs_shape_clone_;
          if (strategy_ != nullptr) {  // if strategy_ is not null, means that check output strategy
            origin_shapes = outputs_shape_clone_;
          }
          MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(stra) << ", shape or divisor "
                        << shape_value << " at " << j << " cannot be divisible by strategy value " << strategy_value
                        << ", shape is " << ShapeToString(origin_shapes[i]) << ", divisor is "
                        << ShapeToString(sub_input_shape);
        } else {
          MS_LOG(ERROR) << name_ << ": Input index " << i << ", the strategy is " << StrategyToString(stra)
                        << ", shape is " << ShapeToString(sub_input_shape) << ", shape value " << shape_value
                        << " at dim index " << j << " cannot be divisible by strategy value " << strategy_value;
        }
        return FAILED;
      }

      if ((LongToUlong(strategy_value) & LongToUlong(strategy_value - 1)) != 0) {
        if ((g_device_manager->DeviceNum() & (g_device_manager->DeviceNum() - 1)) != 0) {
          MS_LOG(INFO) << name_
                       << ": The device num is not the power of 2, thus do not check the strategy as power of 2";
          continue;
        }
        MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(stra)
                      << ", the value of strategy must be the power of 2, but get " << strategy_value;
        return FAILED;
      }
    }
  }

  return SUCCESS;
}

Status OperatorInfo::CheckStrategyValue(const StrategyPtr &strategy, const Shapes &inputs_shape) {
  if (strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null.";
    return FAILED;
  }

  Strategies stra = strategy->GetInputDim();
  return CheckStrategyByVector(stra, inputs_shape);
}

void OperatorInfo::ResetQueueMember() {
  inputs_tensor_info_.clear();
  inputs_tensor_info_new_.clear();
  outputs_tensor_info_.clear();
  outputs_tensor_info_new_.clear();
  outputs_tensor_map_.clear();
  outputs_tensor_map_new_.clear();
  out_dev_matrix_shape_.clear();
  forward_op_.clear();
  mirror_ops_.clear();
  sub_ops_.clear();
  replace_op_.clear();
  replace_op_info_.clear();
  virtual_div_op_.clear();
  if (!is_layout_config_) {
    inputs_tensor_map_.clear();
    inputs_tensor_map_new_.clear();
    dev_matrix_shape_.clear();
  }
  strategy_ = nullptr;
  out_strategy_ = nullptr;
}

Status OperatorInfo::CheckLayoutConfigBase() {
  // size
  if (inputs_tensor_map_.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << name_
                  << ": the size of inputs tensor map and inputs shape must be equal, but the inputs tensor map is "
                  << inputs_tensor_map_ << ", and the inputs shape is " << inputs_shape_;
    return FAILED;
  }

  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    if (inputs_shape_[i].size() != inputs_tensor_map_[i].size()) {
      MS_LOG(ERROR) << name_
                    << ": the size of input tensor map and input shape must be equal, but the inputs tensor map is "
                    << inputs_tensor_map_ << ", and the inputs shape is " << inputs_shape_ << ", the " << i
                    << "th is not equal";
      return FAILED;
    }
  }

  size_t dev_matrix_size = dev_matrix_shape_.size();
  strategy_from_layout_.clear();

  for (size_t j = 0; j < inputs_tensor_map_.size(); ++j) {
    Shape tmp_strategy;
    for (size_t k = 0; k < inputs_tensor_map_[j].size(); ++k) {
      auto map = inputs_tensor_map_[j][k];

      // range
      if (map == MAP_NONE) {
        tmp_strategy.push_back(NO_SPLIT_STRATEGY);
        continue;
      }

      if (map < 0 || map >= SizeToLong(dev_matrix_size)) {
        MS_LOG(ERROR) << name_ << ": the range of tensor_map's value is [-1, " << (dev_matrix_size - 1)
                      << "], but the inputs tensor map is " << inputs_tensor_map_;
        return FAILED;
      }

      // divisible
      auto shard_num = dev_matrix_shape_[dev_matrix_size - LongToSize(map) - 1];
      MS_EXCEPTION_IF_ZERO("shard_num", shard_num);
      if (inputs_shape_[j][k] % shard_num != 0) {
        MS_LOG(ERROR) << name_ << ": the shape is not divisible by layout, the input shape is " << inputs_shape_
                      << ", the dev matrix is " << dev_matrix_shape_ << ", and the tensor map is "
                      << inputs_tensor_map_;
        return FAILED;
      }

      // if shard_num is 1, reset the map to -1
      if (shard_num == NO_SPLIT_STRATEGY) {
        inputs_tensor_map_[j][k] = MAP_NONE;
      }
      tmp_strategy.push_back(shard_num);
    }
    strategy_from_layout_.push_back(tmp_strategy);
  }

  MS_LOG(INFO) << name_ << ": the strategy from layout is " << strategy_from_layout_;
  return SUCCESS;
}

Status OperatorInfo::GetLayoutConfig() {
  auto layout_iter = attrs_.find(LAYOUT);
  if (layout_iter == attrs_.end()) {
    return SUCCESS;
  }

  MS_EXCEPTION_IF_NULL(layout_iter->second);
  auto layout = layout_iter->second;
  if (!layout->isa<ValueDictionary>()) {
    MS_LOG(ERROR) << name_ << ": the layout is not a dict";
    return FAILED;
  }

  auto dict = layout->cast<ValueDictionaryPtr>();
  for (const auto &kv : dict->value()) {
    ValuePtr key_ptr = kv.first;
    ValuePtr value_ptr = kv.second;
    MS_EXCEPTION_IF_NULL(key_ptr);
    MS_EXCEPTION_IF_NULL(value_ptr);
    if (!key_ptr->isa<StringImm>()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": the value of key is not string";
    }

    std::string key = key_ptr->cast<StringImmPtr>()->value();

    if (key == DEV_MATRIX) {
      if (!value_ptr->isa<ValueTuple>()) {
        MS_LOG(ERROR) << name_ << ": the type of dev matrix is not tuple";
        return FAILED;
      }

      dev_matrix_shape_ = GetValue<std::vector<int64_t>>(value_ptr);
      auto used_devices =
        std::accumulate(dev_matrix_shape_.begin(), dev_matrix_shape_.end(), 1, std::multiplies<int64_t>());
      if (used_devices != stage_device_size_) {
        MS_LOG(ERROR) << name_
                      << ": the product of dev matrix must be equal to the stage divece size, but the dev matrix is "
                      << dev_matrix_shape_ << ", but the stage device size is " << stage_device_size_;
        return FAILED;
      }
      continue;
    }

    if (key == INPUT_TENSOR_MAP) {
      auto var = value_ptr->cast<ValueTuplePtr>();
      if (!value_ptr->isa<ValueTuple>()) {
        MS_LOG(ERROR) << name_ << ": the type of input_tensor_map is not tuple";
        return FAILED;
      }

      std::vector<ValuePtr> elements = var->value();
      for (const auto &ele : elements) {
        Shape tensor_map;
        if (ele->isa<ValueSequence>()) {
          auto value_tuple = ele->cast<ValueTuplePtr>();
          std::vector<ValuePtr> value_vector = value_tuple->value();
          (void)std::transform(value_vector.begin(), value_vector.end(), std::back_inserter(tensor_map),
                               [](const ValuePtr &value) { return static_cast<int64_t>(GetValue<int64_t>(value)); });
          inputs_tensor_map_.push_back(tensor_map);
        } else {
          MS_LOG(ERROR) << name_ << ": the format of input tensor map is wrong! Need ValueSequence";
          return FAILED;
        }
      }
      continue;
    }

    MS_LOG(ERROR) << name_ << ": the invalid key for layout: " << key;
    return FAILED;
  }

  MS_LOG(INFO) << name_ << ": the dev matrix is " << dev_matrix_shape_ << ", the tensor map is " << inputs_tensor_map_;

  is_layout_config_ = true;

  return CheckLayoutConfigBase();
}

bool OperatorInfo::IsDynamicShape() {
  for (auto &input_shape : inputs_shape_) {
    auto in_it = std::find_if(input_shape.cbegin(), input_shape.cend(), [&](const int64_t ele) { return ele == -1; });
    if (in_it != input_shape.end()) {
      return True;
    }
  }

  for (auto &output_shape : outputs_shape_) {
    auto out_it =
      std::find_if(output_shape.cbegin(), output_shape.cend(), [&](const int64_t ele) { return ele == -1; });
    if (out_it != output_shape.end()) {
      return True;
    }
  }
  return False;
}

bool OperatorInfo::IsDynamicRank() {
  for (auto &input_shape : inputs_shape_) {
    auto in_it = std::find_if(input_shape.cbegin(), input_shape.cend(), [&](const int64_t ele) { return ele == -2; });
    if (in_it != input_shape.end()) {
      return True;
    }
  }

  for (auto &output_shape : outputs_shape_) {
    auto out_it =
      std::find_if(output_shape.cbegin(), output_shape.cend(), [&](const int64_t ele) { return ele == -2; });
    if (out_it != output_shape.end()) {
      return True;
    }
  }
  return False;
}

bool OperatorInfo::IsSelfDefineShard() {
  bool self_define_shard_value = false;
  auto attr_iter = attrs_.find(parallel::SELF_DEFINE_SHARD);
  if (attr_iter != attrs_.end()) {
    auto self_define_shard = attr_iter->second->cast<BoolImmPtr>();
    MS_EXCEPTION_IF_NULL(self_define_shard);
    self_define_shard_value = self_define_shard->value();
  }
  return self_define_shard_value;
}

Status OperatorInfo::GetRepeatedNumInDevMatrixRight() {
  bool repeated_num_right = true;
  auto iter = attrs_.find(REPEATED_NUM_IN_DEV_MATRIX_RIGHT);
  if (iter != attrs_.end()) {
    MS_EXCEPTION_IF_NULL(iter->second);
    if (iter->second->isa<BoolImm>()) {
      repeated_num_right = iter->second->cast<BoolImmPtr>()->value();
      MS_LOG(INFO) << name_ << ": attr " << REPEATED_NUM_IN_DEV_MATRIX_RIGHT << " will be set to "
                   << repeated_num_right;
    } else {
      MS_LOG(ERROR) << name_ << ": The value of " << REPEATED_NUM_IN_DEV_MATRIX_RIGHT << " is not bool.";
      return FAILED;
    }
  }
  repeated_num_in_dev_matrix_right_ = repeated_num_right;
  return SUCCESS;
}

Status OperatorInfo::InferAttrs() {
  if (infer_attrs_completed_) {
    return SUCCESS;
  }

  if (GetAttrs() != SUCCESS) {
    return FAILED;
  }

  if (GetRepeatedNumInDevMatrixRight() != SUCCESS) {
    return FAILED;
  }

  if (GetLayoutConfig() != SUCCESS) {
    return FAILED;
  }

  if (is_layout_config_ && CheckLayoutConfig() != SUCCESS) {
    return FAILED;
  }

  use_shape_base_ = true;
  self_define_shard_ = IsSelfDefineShard();
  is_dynamic_shape_ = IsDynamicShape();
  is_dynamic_rank_ = IsDynamicRank();
  if (is_dynamic_rank_) {
    MS_LOG(ERROR) << name_
                  << ": it does not support dynamic rank now, the inupts' shape: " << ShapesToString(inputs_shape_)
                  << ", the outputs' shape: " << ShapesToString(outputs_shape_);
    return FAILED;
  }

  inputs_shape_clone_ = inputs_shape_;
  outputs_shape_clone_ = outputs_shape_;

  infer_attrs_completed_ = true;
  return SUCCESS;
}

Status OperatorInfo::InferMirrorOps() {
  mirror_ops_.clear();
  if (inputs_shape_.empty()) {
    MS_LOG(INFO) << name_ << ": The inputs size is empty";
    return SUCCESS;
  }

  if (inputs_tensor_map_.size() != inputs_shape_.size()) {
    MS_LOG(ERROR) << name_ << ": The size of inputs tensor map is not equal to the size of inputs shape";
    return FAILED;
  }

  bool group_is_empty = true;
  for (size_t i = 0; i < inputs_tensor_map_.size(); ++i) {
    std::vector<Group> group;
    if (CreateGroupByTensorMap(inputs_tensor_map_[i], &group) != SUCCESS) {
      ReportError(name_ + ": Create group failed, the input index is " + std::to_string(i));
      mirror_ops_.clear();
      return FAILED;
    }

    OperatorVector mirror_op;
    if (group.empty()) {
      MS_LOG(INFO) << name_ << ": The mirror group is empty, the input index is " << i;
      mirror_ops_.push_back(mirror_op);
      continue;
    }

    group_is_empty = false;
    mirror_op = CreateMirrorOps(group[0].name(), group[0].GetDevNum());
    mirror_ops_.push_back(mirror_op);
  }

  if (group_is_empty) {
    mirror_ops_.clear();
    MS_LOG(INFO) << name_ << ": No need to insert mirror ops";
  }
  return SUCCESS;
}

Status OperatorInfo::InferMirrorOpsByLayout() {
  mirror_ops_.clear();
  if (inputs_shape_.empty()) {
    MS_LOG(INFO) << name_ << ": The inputs size is empty";
    return SUCCESS;
  }

  bool group_is_empty = true;
  for (size_t i = 0; i < inputs_tensor_info_.size(); ++i) {
    auto input_tensor_layout = inputs_tensor_info_[i].tensor_layout();
    auto repeated_rank_list = input_tensor_layout.InferRepeatedGroup();

    OperatorVector mirror_op;
    if (repeated_rank_list.size() == 1) {
      MS_LOG(INFO) << name_ << ": The mirror group is empty, the input index is " << i;
      mirror_ops_.push_back(mirror_op);
      continue;
    }

    Group mirror_group;
    if (g_device_manager->CreateGroup(repeated_rank_list, &mirror_group) != SUCCESS) {
      MS_LOG(ERROR) << name_
                    << ": Create communication group by tensor_map failed, the rank_list is: " << repeated_rank_list
                    << ", the full_name of node is: " << cnode_->fullname_with_scope();
      return FAILED;
    }
    group_is_empty = false;
    mirror_op = CreateMirrorOps(mirror_group.name(), mirror_group.GetDevNum());
    mirror_ops_.push_back(mirror_op);
  }

  if (group_is_empty) {
    mirror_ops_.clear();
    MS_LOG(INFO) << name_ << ": No need to insert mirror ops";
  }
  return SUCCESS;
}

TensorInfoBasePtr CreateTensorInfo(const Shape &device_matrix, const ShapeBasePtr &inputs_shape,
                                   const ShapeBasePtr &inputs_tensor_map) {
  TensorInfoBasePtr out_tensor_info;
  if (inputs_shape->is_list()) {
    std::vector<TensorInfoBasePtr> tensor_info_list;
    for (int64_t i = 0; i < SizeToLong(inputs_shape->size()); ++i) {
      auto tensor_map = inputs_tensor_map->GetElement(i);
      auto shape = inputs_shape->GetElement(i);
      auto input_tensor_info = CreateTensorInfo(device_matrix, shape, tensor_map);
      tensor_info_list.emplace_back(input_tensor_info);
    }
    out_tensor_info = std::make_shared<TensorInfoList>(tensor_info_list);
  } else {
    TensorLayout input_layout;
    input_layout.InitFromVector(device_matrix, inputs_tensor_map->GetValue(), inputs_shape->GetValue());
    TensorInfo input_tensor_info(input_layout);
    out_tensor_info = std::make_shared<TensorInfoValue>(input_tensor_info);
  }
  return out_tensor_info;
}

Status OperatorInfo::InferTensorInfoNew() {
  size_t real_input_index = 0;
  for (size_t i = 0; i < inputs_shape_new_.size(); ++i) {
    // noshape insert default tenosor info
    if (inputs_shape_new_[i]->size() == 0) {
      (void)inputs_tensor_info_new_.emplace_back(std::make_shared<TensorInfoValue>(TensorInfo()));
      continue;
    }
    auto input_tensor_info =
      CreateTensorInfo(dev_matrix_shape_, inputs_shape_new_[i], inputs_tensor_map_new_[real_input_index]);
    inputs_tensor_info_new_.emplace_back(input_tensor_info);
    ++real_input_index;
  }

  for (size_t i = 0; i < outputs_tensor_map_new_.size(); ++i) {
    auto output_tensor_info = CreateTensorInfo(dev_matrix_shape_, outputs_shape_new_[i], outputs_tensor_map_new_[i]);
    outputs_tensor_info_new_.emplace_back(output_tensor_info);
  }
  return SUCCESS;
}

void OperatorInfo::UpdateOutputTensorInfoForInterleaved() {
  if (!std::any_of(inputs_tensor_info_.begin(), inputs_tensor_info_.end(), [](const TensorInfo &input_tensor_info) {
        return input_tensor_info.tensor_layout().IsInterleavedParallel();
      })) {
    return;
  }
  if (std::any_of(outputs_tensor_info_.begin(), outputs_tensor_info_.end(), [](const TensorInfo &output_tensor_info) {
        return output_tensor_info.tensor_layout().IsInterleavedParallel();
      })) {
    return;
  }
  auto interleaved_num = ParallelContext::GetInstance()->fine_grained_micro_interleaved_size();
  auto output_dev_matrix = outputs_tensor_info_[kIndex0].tensor_layout().device_arrangement_origin().array();
  output_dev_matrix[output_dev_matrix.size() - 1] = interleaved_num;
  Arrangement out_device_arrangement_interleaved;
  out_device_arrangement_interleaved.Init(output_dev_matrix);
  auto new_tensor_layout = outputs_tensor_info_[kIndex0].tensor_layout();
  new_tensor_layout.set_device_arrangement_interleaved(out_device_arrangement_interleaved);
  TensorInfo new_output_tensor_info(new_tensor_layout);
  outputs_tensor_info_[kIndex0] = new_output_tensor_info;
}

Status OperatorInfo::InferTensorInfo() {
  if (!inputs_shape_new_.empty()) {
    return InferTensorInfoNew();
  }

  size_t real_input_index = 0;
  for (size_t i = 0; i < inputs_tensor_map_.size(); ++i) {
    // Insert placeholder TensorInfo for optional input
    while (real_input_index < input_value_.size() && input_value_.at(real_input_index) != nullptr &&
           input_value_[real_input_index]->isa<None>()) {
      (void)inputs_tensor_info_.emplace_back(TensorInfo());
      ++real_input_index;
    }
    if (i >= inputs_shape_.size()) {
      (void)inputs_tensor_info_.emplace_back(TensorInfo());
      ++real_input_index;
      continue;
    }
    TensorLayout input_layout;
    if (input_layout.InitFromVector(dev_matrix_shape_, inputs_tensor_map_.at(i), inputs_shape_.at(i)) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer input tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo input_tensor_info(input_layout);
    inputs_tensor_info_.push_back(input_tensor_info);
    ++real_input_index;
  }

  for (size_t i = 0; i < outputs_tensor_map_.size(); ++i) {
    TensorLayout output_layout;
    if (output_layout.InitFromVector(dev_matrix_shape_, outputs_tensor_map_.at(i), outputs_shape_.at(i)) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Infer output tensor layout failed, the index is " << i;
      return FAILED;
    }
    TensorInfo output_tensor_info(output_layout);
    outputs_tensor_info_.push_back(output_tensor_info);
  }

  return SUCCESS;
}

Status OperatorInfo::InferRepeatedCalcInfo() {
  int64_t g_dev_list_size = stage_device_size_;
  int64_t dev_matrix_size =
    std::accumulate(dev_matrix_shape_.begin(), dev_matrix_shape_.end(), 1, std::multiplies<int64_t>());
  if (dev_matrix_size == 0) {
    MS_LOG(ERROR) << name_ << ": The dev matrix size is 0";
    return FAILED;
  }

  if (g_dev_list_size == dev_matrix_size) {
    repeated_calc_num_ = 1;
  } else if (g_dev_list_size % dev_matrix_size == 0) {
    repeated_calc_num_ = g_dev_list_size / dev_matrix_size;
  } else {
    MS_LOG(ERROR) << name_ << ": The strategy is " << StrategyToString(strategy_->GetInputDim()) << ", it requires "
                  << dev_matrix_size << " devices, "
                  << "but the device number of this stage is " << g_dev_list_size << ", it can not be divisible by "
                  << dev_matrix_size;
    return FAILED;
  }
  return SUCCESS;
}

// If repeated calculation, set the repeated_calc_num as the last dimension of dev-matrix in default,
// because if the previous shard is (a, b), and the next shard is (a, 1), adding the repeated_calc_num
// to the last dimension of dev-matrix, there is no need to redistribution.
void OperatorInfo::SetRepeatedCalcDevMatrix() {
  if (repeated_calc_num_ <= 1) {
    return;
  }
  if (repeated_num_in_dev_matrix_right_) {
    dev_matrix_shape_.push_back(repeated_calc_num_);
  } else {
    (void)dev_matrix_shape_.insert(dev_matrix_shape_.cbegin(), repeated_calc_num_);
  }
}

void OperatorInfo::ResetTupleTensorMapIfRepeatedCalc(NewTensorMaps *tensor_map_new) {
  MS_EXCEPTION_IF_NULL(tensor_map_new);
  for (auto &tensor_map : *tensor_map_new) {
    if (tensor_map->is_list()) {
      std::vector<ShapeBasePtr> new_list;
      for (auto &elements : tensor_map->GetAllElements()) {
        std::vector<int64_t> new_shape;
        for (auto &element : elements) {
          if (element != MAP_NONE) {
            element += 1;
          }
          new_shape.emplace_back(element);
        }
        new_list.emplace_back(std::make_shared<ShapeValue>(new_shape));
      }
      tensor_map->set_shape(std::make_shared<ShapeList>(new_list));
    } else {
      std::vector<int64_t> new_shape;
      for (auto &element : tensor_map->GetValue()) {
        if (element != MAP_NONE) {
          element += 1;
        }
        new_shape.emplace_back(element);
      }
      tensor_map->set_shape(std::make_shared<ShapeValue>(new_shape));
    }
  }
}

// If repeated calculation, and the repeated_calc_num is inserted to the last dimension of the dev-matrix,
// the index value of tensor map needs to be increased by 1.
void OperatorInfo::ResetTensorMapIfRepeatedCalc() {
  if ((repeated_calc_num_ <= 1) || !repeated_num_in_dev_matrix_right_) {
    return;
  }

  MS_LOG(DEBUG) << name_ << ": the repeated calc num is " << repeated_calc_num_ << ", and reset the tensor maps";
  for (auto &tensor_map : inputs_tensor_map_) {
    for (auto &element : tensor_map) {
      if (element == MAP_NONE) {
        continue;
      }
      element += 1;
    }
  }

  for (auto &tensor_map : outputs_tensor_map_) {
    for (auto &element : tensor_map) {
      if (element == MAP_NONE) {
        continue;
      }
      element += 1;
    }
  }

  ResetTupleTensorMapIfRepeatedCalc(&inputs_tensor_map_new_);
  ResetTupleTensorMapIfRepeatedCalc(&outputs_tensor_map_new_);
}

// use for loss repeated calculation
Operator CreateVirtualDivOp(int64_t div_num) {
  OperatorName operator_name = VIRTUAL_DIV;
  ValuePtr attr0_value = MakeValue(div_num);
  Attr attr0 = std::make_pair(DIVISOR, attr0_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  return op;
}

Operator CreateDivOp(float scale) {
  OperatorName operator_name = REAL_DIV;
  OperatorAttrs operator_attrs;
  OperatorParams operator_param;
  constexpr size_t parameter_pos = 2;
  mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(scale);
  ValuePtr scale_value = MakeValue(tensor_ptr);
  (void)operator_param.emplace_back(std::make_pair(std::make_pair(Y, scale_value), parameter_pos));
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  return op;
}

Operator CreateScalarFloorDivOp(int64_t div_num) {
  OperatorName operator_name = SCALAR_FLOOR_DIV;
  OperatorAttrs operator_attrs;
  OperatorParams operator_param;
  constexpr size_t parameter_pos = 2;
  ValuePtr scale_value = MakeValue(div_num);
  (void)operator_param.emplace_back(std::make_pair(std::make_pair(Y, scale_value), parameter_pos));
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  return op;
}

Operator CreateScalarDivOp(int64_t div_num) {
  OperatorName operator_name = SCALAR_DIV;
  OperatorAttrs operator_attrs;
  OperatorParams operator_param;
  constexpr size_t parameter_pos = 2;
  ValuePtr scale_value = MakeValue(std::make_shared<Int64Imm>(div_num));
  (void)operator_param.emplace_back(std::make_pair(std::make_pair(Y, scale_value), parameter_pos));
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  return op;
}

Operator CreateScalarMulOp(int64_t scalar) {
  OperatorName operator_name = SCALAR_MUL;
  OperatorAttrs operator_attrs;
  OperatorParams operator_param;
  constexpr size_t parameter_pos = 2;
  ValuePtr scale_value = MakeValue(std::make_shared<Int64Imm>(scalar));
  (void)operator_param.emplace_back(std::make_pair(std::make_pair(Y, scale_value), parameter_pos));
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  return op;
}

static OperatorArgs CreateReduceCommunicationOpArgs(const std::string &reduce_op, const std::string &group) {
  ValuePtr attr0_value = MakeValue(reduce_op);
  ValuePtr attr1_value = MakeValue(group);
  Attr attr0 = std::make_pair(OP, attr0_value);
  Attr attr1 = std::make_pair(GROUP, attr1_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);

  OperatorParams operator_param;
  return std::make_pair(operator_attrs, operator_param);
}

// use for forward all reduce
Operator CreateAllReduceOp(const std::string &reduce_op, const std::string &group) {
  OperatorName operator_name = ALL_REDUCE;
  OperatorArgs operator_arg = CreateReduceCommunicationOpArgs(reduce_op, group);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create all reduce op success, the reduce_op is  " << reduce_op << ", the group is " << group;
  return op;
}

Operator CreateReduceScatterOp(const std::string &reduce_op, const std::string &group) {
  OperatorName operator_name = REDUCE_SCATTER;
  OperatorArgs operator_arg = CreateReduceCommunicationOpArgs(reduce_op, group);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create reduce scatter op success, the reduce_op is  " << reduce_op << ", the group is " << group;
  return op;
}

Operator CreateCastOp(TypePtr type) {
  auto type_id = MakeValue(static_cast<int64_t>(type->type_id()));
  Param param_type = std::make_pair(std::make_pair(DTYPE, type_id), 2);
  OperatorAttrs attrs;
  OperatorParams params = {param_type};
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op_cast = std::make_pair(CAST, args);
  return op_cast;
}

Operator CreateScalarCastOp(TypePtr type) {
  auto type_id = MakeValue(static_cast<int64_t>(type->type_id()));
  Param param_type = std::make_pair(std::make_pair(DTYPE, type_id), 2);
  OperatorAttrs attrs;
  OperatorParams params = {param_type};
  OperatorArgs args = std::make_pair(attrs, params);
  Operator op_cast = std::make_pair(SCALAR_CAST, args);
  return op_cast;
}

int32_t AddCommOpFusionType(const CNodePtr &comm_node, const AnfNodePtr &param_node) {
  MS_EXCEPTION_IF_NULL(comm_node);
  MS_EXCEPTION_IF_NULL(param_node);
  ParameterPtr param;
  if (IsPrimitiveCNode(param_node, prim::kPrimReceive)) {
    const auto param_node_ptr = param_node->user_data<AnfNode>(PIPELINE_PARAM);
    MS_EXCEPTION_IF_NULL(param_node_ptr);
    param = param_node_ptr->cast<ParameterPtr>();
    MS_EXCEPTION_IF_NULL(param);
  } else {
    param = param_node->cast<ParameterPtr>();
  }
  MS_EXCEPTION_IF_NULL(param);
  auto prim = GetValueNode<PrimitivePtr>(comm_node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto attrs = prim->attrs();
  auto param_info = param->param_info();
  int32_t fusion_type = 0;
  if (param_info) {
    fusion_type = param_info->comm_fusion();
  }

  attrs[FUSION] = MakeValue<int64_t>(fusion_type);
  (void)prim->SetAttrs(attrs);
  bool zero3 = ParallelContext::GetInstance()->zero3();
  std::string instance_name = prim->instance_name();
  if (instance_name == PARALLEL_OPTIMIZER_ALLGATHER_NOT_COMPUTE && zero3 &&
      (prim->name() == ALL_GATHER || prim->name() == MICRO_STEP_ALL_GATHER)) {
    prim->set_attr(RECOMPUTE, MakeValue(true));
    prim->set_instance_name(PARALLEL_OPTIMIZER_ALLGATHER);
    auto node_users = comm_node->func_graph()->manager()->node_users();
    auto ag_users = node_users.at(comm_node);
    for (const auto &node_pair : ag_users) {
      if (IsPrimitiveCNode(node_pair.first, prim::kPrimLoad)) {
        auto load_prim = GetCNodePrimitive(node_pair.first->cast<CNodePtr>());
        load_prim->set_attr(RECOMPUTE, MakeValue(true));
        load_prim->set_attr(kAttrParallelOptLoad, MakeValue(true));
      }
    }
    if (IsPrimitiveCNode(comm_node->input(kIndex1), prim::kPrimCast)) {
      auto cast_cnode = comm_node->input(kIndex1)->cast<CNodePtr>();
      auto cast_prim = GetCNodePrimitive(cast_cnode);
      MS_EXCEPTION_IF_NULL(cast_prim);
      cast_prim->set_attr(RECOMPUTE, MakeValue(true));
    }
  }
  MS_LOG(INFO) << "Set comm fusion:" << param->name() << "'s fusion type is " << fusion_type;
  return fusion_type;
}

void AddCommOpMeanFlag(const CNodePtr &comm_node) {
  MS_EXCEPTION_IF_NULL(comm_node);
  auto prim = GetValueNode<PrimitivePtr>(comm_node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto attrs = prim->attrs();
  MS_EXCEPTION_IF_NULL(ParallelContext::GetInstance());
  bool mean_flag = ParallelContext::GetInstance()->gradients_mean();
  attrs[MEAN_FLAG] = MakeValue<bool>(mean_flag);
  (void)prim->SetAttrs(attrs);
}

void AddCNodePrimAttr(const CNodePtr &comm_node, const std::string &attr_name, const ValuePtr &attr_val) {
  MS_EXCEPTION_IF_NULL(comm_node);
  auto prim = GetValueNode<PrimitivePtr>(comm_node->input(0));
  MS_EXCEPTION_IF_NULL(prim);
  auto attrs = prim->attrs();
  attrs[attr_name] = attr_val;
  (void)prim->SetAttrs(attrs);
}

void AddCommOpParamFlag(const CNodePtr &comm_node) {
  MS_EXCEPTION_IF_NULL(comm_node);
  auto graph = comm_node->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_users = manager->node_users()[comm_node->input(1)];
  for (auto &node_user : node_users) {
    if (IsPrimitiveCNode(node_user.first, prim::kPrimSend)) {
      auto prim = GetCNodePrimitive(comm_node);
      (void)prim->AddAttr(PARAMETER_MICRO, MakeValue(0));
      return;
    }
  }
}

Operator CreateAllGatherOp(const std::string &group) {
  OperatorName operator_name = ALL_GATHER;
  // group
  ValuePtr attr0_value = MakeValue(group);
  Attr attr0 = std::make_pair(GROUP, attr0_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create allgather op success, the group is " << group;
  return op;
}

Operator CreateMicroStepAllGatherOp(const std::string &group) {
  bool mean_flag = ParallelContext::GetInstance()->gradients_mean();
  OperatorName operator_name = MICRO_STEP_ALL_GATHER;
  // group
  ValuePtr attr0_value = MakeValue(group);
  Attr attr0 = std::make_pair(GROUP, attr0_value);
  // mean_flag
  ValuePtr attr1_value = MakeValue(mean_flag);
  Attr attr1 = std::make_pair(MEAN_FLAG, attr1_value);
  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);

  OperatorParams operator_param;
  OperatorArgs operator_arg = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create MICRO_STEP_ALL_GATHER success, the group is " << group;
  return op;
}

// use for get tensor slice
Operator CreateGetTensorSliceOp(const TensorLayout &tensor_layout) {
  Shape tensor_map = tensor_layout.tensor_map().array();
  Shape dev_matrix_shape = tensor_layout.device_arrangement().array();
  Shape slice_shape = tensor_layout.base_slice_shape().array();
  Shape full_shape = tensor_layout.tensor_shape().array();
  OperatorName operator_name = GET_TENSOR_SLICE;

  OperatorAttrs attrs;
  ValuePtr dev_mat_value = MakeValue(dev_matrix_shape);
  Param dev_mat_param = std::make_pair(std::make_pair(DEV_MAT, dev_mat_value), 2);
  ValuePtr tensor_map_value = MakeValue(tensor_map);
  Param tensor_map_param = std::make_pair(std::make_pair(TENSOR_MAP, tensor_map_value), 3);
  ValuePtr slice_shape_value = MakeValue(slice_shape);
  Param slice_shape_param = std::make_pair(std::make_pair(SLICE_SHAPE, slice_shape_value), 4);
  ValuePtr full_shape_value = MakeValue(full_shape);
  Param full_shape_param = std::make_pair(std::make_pair(FULL_SHAPE, full_shape_value), 5);
  OperatorParams params = {dev_mat_param, tensor_map_param, slice_shape_param, full_shape_param};
  OperatorArgs operator_arg = std::make_pair(attrs, params);

  Operator op = std::make_pair(operator_name, operator_arg);
  MS_LOG(INFO) << "Create get tensor slice op success, the dev mat and tensor map is "
               << ShapeToString(dev_matrix_shape) << ", " << ShapeToString(tensor_map);
  return op;
}

OperatorVector CreateMirrorOps(const std::string &group_name, size_t dev_num) {
  if (dev_num == 0) {
    MS_LOG(EXCEPTION) << "Invalid dev num: " << dev_num;
  }
  OperatorVector op_for_weight;
  bool mean_flag = ParallelContext::GetInstance()->gradients_mean();
  int64_t grad_accumulation_step = ParallelContext::GetInstance()->grad_accumulation_step();
  int64_t split_stage_num = ParallelContext::GetInstance()->pipeline_stage_split_num();

  ValuePtr attr0_value = MakeValue(group_name);
  ValuePtr attr1_value = MakeValue(SizeToLong(dev_num));
  ValuePtr attr2_value = MakeValue(mean_flag);

  Attr attr0 = std::make_pair(GROUP, attr0_value);
  Attr attr1 = std::make_pair(DEV_NUM, attr1_value);
  Attr attr2 = std::make_pair(MEAN_FLAG, attr2_value);

  OperatorAttrs operator_attrs;
  operator_attrs.push_back(attr0);
  operator_attrs.push_back(attr1);
  operator_attrs.push_back(attr2);

  OperatorName operator_name;
  if (grad_accumulation_step > 1 || split_stage_num > 1) {
    operator_name = MIRROR_MICRO_STEP_OPERATOR;
  } else {
    operator_name = MIRROR_OPERATOR;
  }

  OperatorParams operator_param;
  OperatorArgs operator_args = std::make_pair(operator_attrs, operator_param);

  Operator op = std::make_pair(operator_name, operator_args);

  op_for_weight.push_back(op);
  MS_LOG(INFO) << "The group name is " << group_name << ", the dev num is " << dev_num << ", the mean flag is "
               << mean_flag;
  return op_for_weight;
}

Status OperatorInfo::CreateGroupByTensorMap(const Shape &tensor_map, std::vector<Group> *group) {
  if (group == nullptr) {
    MS_LOG(ERROR) << name_ << ": The group is null.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);
  RankList group_devices;
  if (dev_matrix.GetDevicesByTensorMap(tensor_map, &group_devices) != SUCCESS) {
    return FAILED;
  }

  if (group_devices.size() == 1 && !((ParallelContext::GetInstance()->grad_accumulation_step() > 1 ||
                                      ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) &&
                                     ParallelContext::GetInstance()->enable_parallel_optimizer())) {
    MS_LOG(INFO) << name_ << ": The dev size is 1, no need to create group.";
    return SUCCESS;
  }

  Group g;
  if (g_device_manager->CreateGroup(group_devices, &g) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create communication group by tensor_map failed, the rank_list is: " << group_devices
                  << ", the input strategy is " << strategy_->GetInputDim()
                  << ", the full_name of node is: " << cnode_->fullname_with_scope();
    return FAILED;
  }
  group->push_back(g);
  return SUCCESS;
}

Status OperatorInfo::CreateGroupForOptShard(TensorLayout *tensor_layout, std::vector<Group> *groups) {
  if (groups == nullptr) {
    MS_LOG(ERROR) << name_ << ": The group is null.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, tensor_layout->device_arrangement_origin().array());
  RankList group_devices;
  Shape tensor_map = tensor_layout->origin_tensor_map().array();
  if (dev_matrix.GetDevicesByTensorMap(tensor_map, &group_devices) != SUCCESS) {
    return FAILED;
  }

  if (group_devices.size() == 1) {
    MS_LOG(INFO) << name_ << ": The dev size is 1, no need to create group.";
    return SUCCESS;
  }
  int64_t repeated_size = SizeToLong(group_devices.size());
  int64_t optimizer_weight_shard_size = ParallelContext::GetInstance()->optimizer_weight_shard_size();
  MS_EXCEPTION_IF_ZERO("optimizer_weight_shard_size", optimizer_weight_shard_size);
  if (optimizer_weight_shard_size != -1 && repeated_size > optimizer_weight_shard_size) {
    // not fully use opt shard
    int64_t index = std::find(group_devices.begin(), group_devices.end(), rank) - group_devices.begin();
    if (repeated_size % optimizer_weight_shard_size != 0 || repeated_size < optimizer_weight_shard_size) {
      MS_LOG(WARNING) << "Parallel optimizer:"
                      << " optimizer_weight_shard_size " << optimizer_weight_shard_size
                      << " can not be applied for the parameter used by" << cnode_->fullname_with_scope()
                      << " The data parallel group size is " << repeated_size;
      return FAILED;
    }
    repeated_size = repeated_size / optimizer_weight_shard_size;
    // create allgather group
    // eg: optimizer_weight_shard_size = 2, [0, 8, 16, 24] -> [0, 8], [16, 24]
    RankList new_group_devices(
      group_devices.begin() + index / optimizer_weight_shard_size * optimizer_weight_shard_size,
      group_devices.begin() + (index / optimizer_weight_shard_size + 1) * optimizer_weight_shard_size);
    Group allgather_group;
    if (g_device_manager->CreateGroup(new_group_devices, &allgather_group) != SUCCESS) {
      MS_LOG(ERROR) << name_
                    << ": Create communication group for allgather in optimizer parallel failed,"
                       " the rank_list is: "
                    << group_devices << ", the input strategy is " << strategy_->GetInputDim()
                    << ", the full_name of node is: " << cnode_->fullname_with_scope();
      return FAILED;
    }
    groups->push_back(allgather_group);
    tensor_layout->set_opt_shard_group(allgather_group.name());
    MS_LOG(INFO) << name_ << ": Parallel optimizer, create allgather group " << allgather_group.name();
    // create mirror group
    // eg: optimizer_weight_shard_size = 2, [0, 8, 16, 24] -> [0, 16], [8, 24]
    int64_t device_num = g_device_manager->stage_device_num();
    MS_EXCEPTION_IF_ZERO("repeated_size", repeated_size);
    Shape dev_mat = {repeated_size, device_num / repeated_size};
    DeviceMatrix temp_dev_matrix(rank, stage_device_list_, dev_mat);
    RankList mirror_group_devices;
    if (temp_dev_matrix.GetDevicesAlongDim(0, &mirror_group_devices) != SUCCESS) {
      return FAILED;
    }
    Group mirror_group;
    if (g_device_manager->CreateGroup(mirror_group_devices, &mirror_group) != SUCCESS) {
      MS_LOG(ERROR) << name_
                    << ": Create communication group for mirror in optimizer parallel failed,"
                       " the rank_list is: "
                    << group_devices << ", the input strategy is " << strategy_->GetInputDim()
                    << ", the full_name of node is: " << cnode_->fullname_with_scope();
      return FAILED;
    }
    groups->push_back(mirror_group);
    tensor_layout->set_opt_shard_mirror_group(mirror_group.name());
    MS_LOG(INFO) << name_ << ": Parallel optimizer, create mirror group " << mirror_group.name();
  } else {
    // fully use opt shard
    // create allgather group
    Group allgather_group;
    if (g_device_manager->CreateGroup(group_devices, &allgather_group) != SUCCESS) {
      MS_LOG(ERROR) << name_
                    << ": Create communication group for allgather in optimizer parallel failed,"
                       " the rank_list is: "
                    << group_devices << ", the input strategy is " << strategy_->GetInputDim()
                    << ", the full_name of node is: " << cnode_->fullname_with_scope();
      return FAILED;
    }
    groups->push_back(allgather_group);
    tensor_layout->set_opt_shard_group(allgather_group.name());
    MS_LOG(INFO) << name_ << ": Parallel optimizer, create allgather group " << allgather_group.name();
  }
  // save in tensor_layout for strategy ckpt
  auto integrated_save = ParallelContext::GetInstance()->optimizer_weight_shard_aggregated_save();
  if (!integrated_save) {
    tensor_layout->set_opt_weight_shard_size(LongToInt(optimizer_weight_shard_size));
    if (optimizer_weight_shard_size > 0 && group_devices.size() < LongToSize(optimizer_weight_shard_size)) {
      tensor_layout->set_opt_weight_shard_size(SizeToInt(group_devices.size()));
    }
    MS_EXCEPTION_IF_ZERO("SizeToLong(group_devices.size()) - 1", SizeToLong(group_devices.size()) - 1);
    int64_t opt_weight_shard_step =
      (group_devices.back() - group_devices.front()) / (SizeToLong(group_devices.size()) - 1);
    tensor_layout->set_opt_weight_shard_step(LongToInt(opt_weight_shard_step));
    MS_LOG(INFO) << name_ << "Parallel optimizer, save opt_weight_shard_step " << opt_weight_shard_step
                 << " in strategy ckpt";
  }
  return SUCCESS;
}

static void InsertDivOpToNodeInput(const CNodePtr &node, int64_t div_num, size_t index, const string &instance_name) {
  MS_EXCEPTION_IF_NULL(node);
  FuncGraphPtr func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  // instance the div operator
  Operator div_op = CreateScalarFloorDivOp(div_num);

  // Insert it as the input of the node
  AnfNodePtr input = node->input(index);
  MS_EXCEPTION_IF_NULL(input);
  InsertNode(div_op, node, index, node->input(index), func_graph, instance_name);
}

void OperatorInfo::ChangeMakeTupleConstant(const CNodePtr &cnode, size_t make_tuple_index) {
  if (!IsPrimitiveCNode(cnode->input(make_tuple_index), prim::kPrimMakeTuple)) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << ": the dst shape is not make tuple";
  }
  size_t input_dim = inputs_shape_[0].size();
  auto shard_size = strategy_->GetInputDim()[0];
  if (input_dim != shard_size.size()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode) << name_ << ": the input dim is " << input_dim
                                       << ", but the size of strategy is " << shard_size.size();
  }

  auto make_tuple = cnode->input(make_tuple_index);
  auto make_tuple_cnode = make_tuple->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(make_tuple_cnode);
  size_t diff_len = outputs_shape_[0].size() - input_dim;  // right align for BroadcastTo
  for (size_t i = 0; i < input_dim; ++i) {
    if (shard_size[i] <= 1) {
      continue;
    }
    auto value_node = GetValueNode(make_tuple_cnode->input(i + diff_len + 1));
    if (value_node == nullptr) {
      std::string instance_name = name_ + "div";
      InsertDivOpToNodeInput(make_tuple_cnode, shard_size[i], i + diff_len + 1, instance_name);
    } else if (value_node->isa<Int64Imm>()) {
      MS_EXCEPTION_IF_ZERO("shard_size", shard_size[i]);
      auto origin_value = GetValue<int64_t>(value_node);
      if (origin_value < 0) {  // such as BroadcastTo, the dst_shape maybe has -1
        continue;
      }
      if (origin_value % shard_size[i] != 0) {
        MS_LOG_WITH_NODE(EXCEPTION, make_tuple_cnode)
          << name_ << ": the origin value is " << origin_value << ", can not be div by shard size " << shard_size[i]
          << ", the make tuple index of this op is " << make_tuple_index << ", the input index of make_tuple is "
          << (i + diff_len + 1);
      }
      int64_t replace_value = GetValue<int64_t>(value_node) / shard_size[i];
      auto replace_value_ptr = MakeValue(replace_value);
      auto replace_value_node = std::make_shared<ValueNode>(replace_value_ptr);
      auto manager = make_tuple->func_graph()->manager();
      manager->SetEdge(make_tuple, i + diff_len + 1, replace_value_node);
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, make_tuple_cnode)
        << name_ << ": the input of make_tuple is value node but not int64, the index is " << (i + 1);
    }
  }
}

Status OperatorInfo::CreateGroupByDim(size_t axis, std::vector<Group> *group) {
  if (group == nullptr) {
    MS_LOG(ERROR) << name_ << ": The group is null.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  int64_t rank = g_device_manager->global_rank();
  DeviceMatrix dev_matrix(rank, stage_device_list_, dev_matrix_shape_);

  return CreateGroupByDimWithDevMatrix(&dev_matrix, axis, group);
}

Status OperatorInfo::CreateGroupByDimWithDevMatrix(DeviceMatrix *dev_matrix, size_t axis, std::vector<Group> *group) {
  if (group == nullptr) {
    MS_LOG(ERROR) << name_ << ": The group is null.";
    return FAILED;
  }
  CheckGlobalDeviceManager();
  RankList group_devices;
  if (dev_matrix->GetDevicesAlongDim(SizeToUlong(axis), &group_devices) != SUCCESS) {
    return FAILED;
  }

  if (group_devices.size() == 1) {
    MS_LOG(INFO) << name_ << ": The dev size is 1, no need to create group.";
    return SUCCESS;
  }
  Group g;
  if (g_device_manager->CreateGroup(group_devices, &g) != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Create communication group by dim failed, the rank_list is: " << group_devices
                  << ", the input strategy is " << strategy_->GetInputDim()
                  << ", the full_name of node is: " << cnode_->fullname_with_scope();
    return FAILED;
  }
  MS_LOG(INFO) << name_ << ": Create communication group by dim " << axis
               << " success, the rank_list is: " << group_devices;
  group->push_back(g);
  return SUCCESS;
}

Shape GetSliceShape(const Shape &tensor_shape, const Dimensions &strategy) {
  Shape slice_shape;
  if (std::any_of(strategy.begin(), strategy.end(), [](int64_t value) { return value <= 0; })) {
    MS_LOG(ERROR) << "Invalid strategy: " << ShapeToString(strategy) << ", the element is less than or equal to 0";
    return slice_shape;
  }
  for (size_t i = 0; i < strategy.size(); ++i) {
    slice_shape.push_back(tensor_shape.at(i) / strategy.at(i));
  }
  return slice_shape;
}

Status OperatorInfo::Init(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy,
                          const std::vector<std::shared_ptr<TensorLayout>> &in_tensor_layouts,
                          const std::vector<std::shared_ptr<TensorLayout>> &out_tensor_layouts) {
  if (!in_tensor_layouts.empty() || !out_tensor_layouts.empty()) {
    return InitWithTensorLayout(in_tensor_layouts, out_tensor_layouts);
  }

  if (InitWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status OperatorInfo::Init(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy,
                          const std::vector<TensorLayoutBasePtr> &in_tensor_layouts,
                          const std::vector<TensorLayoutBasePtr> &out_tensor_layouts) {
  if (!in_tensor_layouts.empty()) {
    return InitWithTensorLayoutForNewShape(in_tensor_layouts, out_tensor_layouts);
  }

  if (InitWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    MS_LOG(ERROR) << name_ << " : Init failed.";
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init success.";
  return SUCCESS;
}

Status OperatorInfo::InitForCostModel(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) {
  if (InitForCostModelWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    ReportError(name_ + " : Init for cost model failed.");
    return FAILED;
  }

  MS_LOG(INFO) << name_ << " : Init for cost model success.";
  return SUCCESS;
}

void OperatorInfo::DivisorsReplaceShapes() {
  if (!dynamic_shape_flag_) {
    return;
  }

  inputs_shape_ = inputs_divisor_;
  outputs_shape_ = outputs_divisor_;
}

void OperatorInfo::ResumeShapes() {
  if (!dynamic_shape_flag_) {
    return;
  }

  inputs_shape_ = inputs_shape_clone_;
  outputs_shape_ = outputs_shape_clone_;
}

void OperatorInfo::DynamicShapeCheckStrategyLog() {
  if (!dynamic_shape_flag_) {
    return;
  }
  MS_LOG(ERROR) << name_ << ": the origin shape of inputs is " << ShapesToString(inputs_shape_clone_)
                << ", but the divisor info of inputs is " << ShapesToString(inputs_divisor_);
}

// auto insert repeated_calculation_num for dev_matrix_shape when repeated_calculation_num > 1
Status OperatorInfo::InitForCostModelWithAutoRepeatCalc(const StrategyPtr &in_strategy,
                                                        const StrategyPtr &out_strategy) {
  if (!is_layout_config_ && in_strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The strategy is null, the inputs shape is " << inputs_shape_;
    return FAILED;
  }

  // need to clear queues before Init(),
  // because Init() may be called multiple times by cost model
  ResetQueueMember();

  if (InferAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferAttrs failed.";
    return FAILED;
  }

  // if layout is configured, no need to check strategy and infer dev matrix
  if (!is_layout_config_) {
    DivisorsReplaceShapes();  // in dynamic shape, using divisors replace to shapes before CheckStrategy
    // must be after InferAttrs()
    if (CheckStrategy(in_strategy) != SUCCESS) {
      DynamicShapeCheckStrategyLog();
      FILTER_LOG(is_auto_parallel_) << name_ << ": CheckStrategy failed.";
      return FAILED;
    }
    ResumeShapes();  // in dynamic shape, resume shapes after CheckStrategy

    if (is_dynamic_shape_ && CheckStrategyForDynamicShape(in_strategy) != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": Check strategy for dynamic shape failed";
      return FAILED;
    }
    strategy_ = in_strategy;

    set_out_strategy(out_strategy);
    if (out_strategy && CheckOutputStrategy(out_strategy) != SUCCESS) {
      if (is_in_layout_propagation_) {
        MS_LOG(INFO) << name_ << ": The output strategy is invalid";
        return FAILED;
      }
      MS_LOG(ERROR) << name_ << ": The output strategy is invalid";
      return FAILED;
    }

    if (InferDevMatrixShape() != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": InferDevMatrixShape failed.";
      return FAILED;
    }

    used_devices_ = std::accumulate(dev_matrix_shape_.begin(), dev_matrix_shape_.end(), 1, std::multiplies<int64_t>());

    // must be after InferDevMatrixShape
    if (InferRepeatedCalcInfo() != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": InferRepeatedCalcInfo failed.";
      return FAILED;
    }

    // if repeated calculation, need to set the repeated_calc_num as the last dimension of dev-matrix for layout
    SetRepeatedCalcDevMatrix();

    if (InferTensorMap() != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": InferTensorMap failed.";
      return FAILED;
    }

    ResetTensorMapIfRepeatedCalc();
  } else {
    if (InferOutputTensorMap() != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": InferOutputTensorMap failed.";
      return FAILED;
    }
  }

  if (InferTensorInfo() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferTensorInfo failed.";
    return FAILED;
  }
  auto stage_dev_num = LongToSize(g_device_manager->stage_device_num());
  if ((stage_dev_num & (stage_dev_num - 1)) == 0) {
    return SUCCESS;
  }
  if (InferForwardCommunication() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": InferForwardCommunication failed in auto parallel searching strategies step.";
    return FAILED;
  }

  if (InferMirrorOps() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": InferMirrorOps failed in auto parallel searching strategies step.";
    return FAILED;
  }
  return SUCCESS;
}

Status OperatorInfo::InitWithAutoRepeatCalc(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) {
  if (!is_layout_config_ && in_strategy == nullptr) {
    MS_LOG(ERROR) << name_ << ": The input strategy is null.";
    return FAILED;
  }

  if (InitForCostModelWithAutoRepeatCalc(in_strategy, out_strategy) != SUCCESS) {
    return FAILED;
  }

  if (InferForwardCommunication() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferForwardCommunication failed.";
    return FAILED;
  }

  if (InferMirrorOps() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferMirrorOps failed.";
    return FAILED;
  }

  if (InferVirtualDivOps() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferVirtualDivOps failed.";
    return FAILED;
  }

  InferReplaceOps();
  return SUCCESS;
}

Status OperatorInfo::CheckInputLayout() {
  if (is_in_layout_propagation_) {
    MS_LOG(INFO)
      << "Current op " << name_
      << " does not support config layout. Please check "
         "https://www.mindspore.cn/docs/zh-CN/master/api_python/operator_list_parallel.html to get limitation "
         "and more details";
  } else {
    MS_LOG(ERROR)
      << "Current op " << name_
      << " does not support config layout. Please check "
         "https://www.mindspore.cn/docs/zh-CN/master/api_python/operator_list_parallel.html to get limitation "
         "and more details";
  }
  // Check self_define_shard attribute
  if (!self_define_shard_) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << "Please set add_prim_attr('self_define_shard', True) to " << name_
                   << " to config layout for this ops";
    } else {
      MS_LOG(ERROR) << "Please set add_prim_attr('self_define_shard', True) to " << name_
                    << " to config layout for this ops";
    }
    return FAILED;
  }
  return FAILED;
}

Status OperatorInfo::InitWithTensorLayout(const std::vector<std::shared_ptr<TensorLayout>> &in_tensor_layouts,
                                          const std::vector<std::shared_ptr<TensorLayout>> &out_tensor_layouts) {
  ResetQueueMember();

  if (InferAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferAttrs failed.";
    return FAILED;
  }

  size_t real_input_index = 0;
  for (const auto &input_layout : in_tensor_layouts) {
    // Insert placeholder TensorInfo for optional input
    while (real_input_index < input_value_.size() && input_value_[real_input_index] != nullptr &&
           input_value_[real_input_index]->isa<None>()) {
      (void)inputs_tensor_info_.emplace_back(TensorInfo());
      ++real_input_index;
    }
    TensorInfo input_tensor_info(*input_layout);
    inputs_tensor_info_.push_back(input_tensor_info);
    ++real_input_index;
  }
  DivisorsReplaceShapes();
  if (CheckInputLayout() != SUCCESS) {
    if (CheckShardingPropagation()) {
      MS_LOG(INFO) << name_ << ": CheckInputLayout failed.";
    } else {
      MS_LOG(ERROR) << name_ << ": CheckInputLayout failed.";
    }
    return FAILED;
  }
  ResumeShapes();
  for (const auto &output_layout : out_tensor_layouts) {
    TensorInfo output_tensor_info(*output_layout);
    outputs_tensor_info_.push_back(output_tensor_info);
  }

  if (outputs_tensor_info_.size() != outputs_shape_.size()) {
    outputs_tensor_info_.clear();
    // Need be override
    if (InferOutputTensorInfo() != SUCCESS) {
      if (is_in_layout_propagation_) {
        MS_LOG(INFO) << name_ << ": InferOutputTensorLayout failed.";
      } else {
        MS_LOG(ERROR) << name_ << ": InferOutputTensorLayout failed.";
      }
      return FAILED;
    }
  }

  if (outputs_tensor_info_.size() != outputs_shape_.size()) {
    if (is_in_layout_propagation_) {
      MS_LOG(INFO) << name_ << ": the output tensor layout num " << outputs_tensor_info_.size()
                   << " does not match the output num " << outputs_shape_.size();
    } else {
      MS_LOG(ERROR) << name_ << ": the output tensor layout num " << outputs_tensor_info_.size()
                    << " does not match the output num " << outputs_shape_.size();
    }
    return FAILED;
  }

  DivisorsReplaceShapes();
  if (CheckOutputLayout() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": CheckLayout failed.";
    return FAILED;
  }
  ResumeShapes();

  if (InitWithTensorLayoutPostProcess() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InitWithTensorLayoutPostProcess failed.";
    return FAILED;
  }
  return SUCCESS;
}

Status OperatorInfo::InitWithTensorLayoutPostProcess() {
  // Need be override
  if (InferForwardCommunicationByLayout() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferForwardCommunication failed.";
    return FAILED;
  }

  if (InferMirrorOpsByLayout() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferMirrorOps failed.";
    return FAILED;
  }
  if (InferVirtualDivOpsByLayout() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferVirtualDivOps failed.";
    return FAILED;
  }

  InferReplaceOps();
  return SUCCESS;
}

Status OperatorInfo::InitWithTensorLayoutForNewShape(const std::vector<TensorLayoutBasePtr> &in_tensor_layouts,
                                                     const std::vector<TensorLayoutBasePtr> &out_tensor_layouts) {
  ResetQueueMember();
  if (InferAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferAttrs failed.";
    return FAILED;
  }

  size_t real_input_index = 0;
  for (const auto &input_layout : in_tensor_layouts) {
    // Insert placeholder TensorInfo for optional input
    if (real_input_index < input_value_.size() && input_value_[real_input_index] != nullptr &&
        input_value_[real_input_index]->isa<None>()) {
      (void)inputs_tensor_info_new_.emplace_back(std::make_shared<TensorInfoValue>(TensorInfo()));
      ++real_input_index;
      continue;
    }
    if (input_layout->no_shape_layout()) {
      if (input_layout->is_list()) {
        std::vector<TensorInfoBasePtr> info_list(input_layout->size(), std::make_shared<TensorInfoValue>(TensorInfo()));
        inputs_tensor_info_new_.emplace_back(std::make_shared<TensorInfoList>(info_list));
      } else {
        inputs_tensor_info_new_.emplace_back(std::make_shared<TensorInfoValue>(TensorInfo()));
      }
    } else if (input_layout->is_list()) {
      inputs_tensor_info_new_.emplace_back(std::make_shared<TensorInfoList>(input_layout));
    } else {
      inputs_tensor_info_new_.emplace_back(std::make_shared<TensorInfoValue>(input_layout));
    }
    ++real_input_index;
  }
  DivisorsReplaceShapes();
  if (CheckInputLayout() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": CheckInputLayout failed.";
    return FAILED;
  }
  ResumeShapes();
  for (const auto &output_layout : out_tensor_layouts) {
    if (output_layout->is_list()) {
      outputs_tensor_info_new_.emplace_back(std::make_shared<TensorInfoList>(output_layout));
    } else {
      outputs_tensor_info_new_.emplace_back(std::make_shared<TensorInfoValue>(output_layout));
    }
  }

  if (outputs_tensor_info_new_.size() != outputs_shape_new_.size()) {
    outputs_tensor_info_new_.clear();
    // Need be override
    if (InferOutputTensorInfo() != SUCCESS) {
      MS_LOG(ERROR) << name_ << ": InferOutputTensorLayout failed.";
      return FAILED;
    }
  }

  if (outputs_tensor_info_new_.size() != outputs_shape_new_.size()) {
    MS_LOG(ERROR) << name_ << ": the output tensor layout num " << outputs_tensor_info_new_.size()
                  << " does not match the output num " << outputs_shape_new_.size();
    return FAILED;
  }

  DivisorsReplaceShapes();
  if (CheckOutputLayout() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": CheckLayout failed.";
    return FAILED;
  }
  ResumeShapes();

  if (InitWithTensorLayoutPostProcess() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InitWithTensorLayoutPostProcess failed.";
    return FAILED;
  }
  return SUCCESS;
}

std::vector<std::shared_ptr<Edge>> OperatorInfo::GetAliveSuccEdges() {
  std::vector<std::shared_ptr<Edge>> ret;
  for (auto &edge : succ_edges_) {
    if ((edge->next_operator()->is_alive()) && (edge->next_operator()->name().find(RELU) != std::string::npos)) {
      ret.push_back(edge);
    } else if ((edge->next_operator()->is_alive()) && (edge->next_operator()->name().find(CAST) != std::string::npos)) {
      // CAST is ordered in front of L2NORMALIZE
      ret.push_back(edge);
    }
  }
  for (auto &edge : succ_edges_) {
    if ((edge->next_operator()->is_alive()) && (edge->next_operator()->name().find(RELU) == std::string::npos) &&
        (edge->next_operator()->name().find(CAST) == std::string::npos)) {
      ret.push_back(edge);
    }
  }
  return ret;
}

std::vector<std::shared_ptr<Edge>> OperatorInfo::GetAlivePrevEdges() {
  std::vector<std::shared_ptr<Edge>> ret;
  for (auto &edge : prev_edges_) {
    if (edge->prev_operator()->is_alive()) {
      ret.push_back(edge);
    }
  }
  return ret;
}

void OperatorInfo::ReplacePreEdge(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplacePreEdge: the op is null.";
    return;
  }
  for (auto &edge : prev_edges_) {
    if (edge->prev_operator() == op) {
      edge = new_edge;
      return;
    }
  }
  MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Replace edge failed: no edge has been replaced";
}

void OperatorInfo::ReplaceSuccEdge(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplaceSuccEdge: the op is null.";
    return;
  }
  for (auto &edge : succ_edges_) {
    if (edge->next_operator() == op) {
      edge = new_edge;
      return;
    }
  }
  MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Replace edge failed: no edge has been replaced";
}

void OperatorInfo::ReplacePreEdges(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplacePreEdges: the op is null.";
    return;
  }
  std::vector<std::shared_ptr<Edge>> update_pre_edges;
  for (auto &edge : prev_edges_) {
    if (edge->prev_operator() != op) {
      update_pre_edges.push_back(edge);
    }
  }
  update_pre_edges.push_back(new_edge);
  prev_edges_ = update_pre_edges;
}

void OperatorInfo::ReplaceSuccEdges(const std::shared_ptr<OperatorInfo> &op, const std::shared_ptr<Edge> &new_edge) {
  if (op == nullptr) {
    MS_LOG(ERROR) << name_ << ": ReplaceSuccEdges: the op is null";
    return;
  }
  std::vector<std::shared_ptr<Edge>> update_pre_edges;
  for (auto &edge : succ_edges_) {
    if (edge->next_operator() != op) {
      update_pre_edges.push_back(edge);
    }
  }
  update_pre_edges.push_back(new_edge);
  succ_edges_ = update_pre_edges;
}

std::shared_ptr<Strategies> OperatorInfo::GenerateBatchStrategiesWithCheck() {
  if (InferAttrs() != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Infer attrs failed";
  }
  DivisorsReplaceShapes();  // in dynamic shape, using divisors replace to shapes before GenerateBatchStrategies

  std::shared_ptr<Strategies> batch_strategy = GenerateBatchStrategies();
  if (batch_strategy->size() != inputs_shape_.size()) {
    MS_LOG(WARNING) << "The inputs size:" << inputs_shape_.size()
                    << " is not equal to the generated batch parallel strategies size:" << batch_strategy->size();
    return batch_strategy;
  }
  int64_t shard_size = g_device_manager->stage_device_num();
  std::vector<std::pair<size_t, size_t>> changed_pos;
  for (size_t i = 0; i < inputs_shape_.size(); ++i) {
    auto stra = batch_strategy->at(i);
    auto input_shape = inputs_shape_.at(i);
    if (stra.size() != input_shape.size()) {
      MS_LOG(WARNING) << "The " << i << " input size:" << input_shape.size() << " is not equal to the " << i
                      << " generated batch parallel strategy size:" << stra.size();
      return batch_strategy;
    }
    for (size_t j = 0; j < input_shape.size(); ++j) {
      if (stra[j] == 1) {
        continue;
      }
      if (stra[j] != g_device_manager->stage_device_num()) {
        MS_LOG(WARNING) << "The batch parallel value is not equal to device num, skip adjust it.";
        return batch_strategy;
      }
      shard_size = std::gcd(input_shape[j], shard_size);
      changed_pos.push_back({i, j});
    }
  }
  for (auto &pair : changed_pos) {
    batch_strategy->at(pair.first).at(pair.second) = shard_size;
  }

  ResumeShapes();
  return batch_strategy;
}

std::shared_ptr<Strategies> GenerateBatchStrategiesBySplitFlag(const Shapes &shapes,
                                                               const std::vector<bool> &split_flag_list) {
  if (shapes.size() != split_flag_list.size()) {
    MS_LOG(ERROR) << "Split_flag_list do not have the same size as inputs shape, " << split_flag_list.size() << " : "
                  << shapes.size();
    return nullptr;
  }
  CheckGlobalDeviceManager();
  int64_t dev_num = g_device_manager->stage_device_num();
  Strategies strategy_v;
  for (size_t i = 0; i != shapes.size(); i++) {
    if (shapes[i].empty()) {
      MS_LOG(INFO) << "Elements of shapes is empty.";
      Dimensions empty_element;
      strategy_v.push_back(empty_element);
    } else {
      Dimensions element(shapes[i].size(), 1);
      if (split_flag_list[i]) {
        element[0] = dev_num;
      }
      strategy_v.push_back(element);
    }
  }
  return std::make_shared<Strategies>(strategy_v);
}

void OperatorInfo::ReComputeBatchSplitFlagList() {
  if (!inputs_shape_.empty()) {
    split_flag_list_[0] = true;
  }
}

void OperatorInfo::ComputeBatchSplitFlagList() {
  split_flag_list_.clear();
  for (auto iter = inputs_shape_.begin(); iter != inputs_shape_.end(); ++iter) {
    split_flag_list_.push_back(false);
  }
  ReComputeBatchSplitFlagList();
}

// This is a common method for checking whether the generated strategy has the correct number of devuces.
Status PrepareStrategyBase(int64_t stage_id, size_t dev_num, const Shapes &inputs_partitions, StrategyPtr *const sp) {
  if (sp == nullptr) {
    MS_LOG(ERROR) << "The strategy is null.";
    return FAILED;
  }
  int64_t product = 1;

  for (auto &input_partition : inputs_partitions) {
    product *= std::accumulate(input_partition.begin(), input_partition.end(), 1, std::multiplies<int64_t>());
  }
  const auto fully_use_device = CostModelContext::GetInstance()->fully_use_device();
  if (!fully_use_device) {
    if (LongToSize(product) > dev_num) {
      return FAILED;
    }
  } else {
    if ((product != 1) && (LongToSize(product) != dev_num)) {
      return FAILED;
    }
  }
  Strategies stras(inputs_partitions);
  (*sp) = std::make_shared<Strategy>(stage_id, stras);
  return SUCCESS;
}

std::shared_ptr<Strategies> OperatorInfo::GenerateBatchStrategies() {
  if (InferAttrs() != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Infer attrs failed";
  }
  ComputeBatchSplitFlagList();
  return GenerateBatchStrategiesBySplitFlag(inputs_shape_, split_flag_list_);
}

// generate strategies for that each dimension of input0 and input1 is relevant, such as: ([a, b, c, d], [a, b, c, d])
Status GenerateStrategiesForTwoEqualInputs(int64_t stage_id, const Shapes &inputs_shape,
                                           const Shapes &splittable_inputs, std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if ((inputs_shape.size() != 2) || (splittable_inputs.size() != 2)) {
    MS_LOG(ERROR) << "The inputs size is wrong.";
    return FAILED;
  }

  if ((inputs_shape[0].size() != inputs_shape[1].size()) ||
      (splittable_inputs[0].size() != splittable_inputs[1].size())) {
    MS_LOG(ERROR) << "The size of two inputs are not equal.";
    return FAILED;
  }

  Shapes input0_shape = {inputs_shape[0]};
  Shapes input0_splittable = {splittable_inputs[0]};
  if (GenerateStrategiesForIndependentInputs(stage_id, input0_shape, input0_splittable, sp_vector) != SUCCESS) {
    return FAILED;
  }

  for (auto &sp : *sp_vector) {
    sp->ExpandInputDimFromOneToTwo();
  }

  return SUCCESS;
}

// generate strategies for that input0 and input1 have relevant dimensions, and input0 needs to broadcast
// such as: ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
Status GenerateStrategiesForBroadcastLeft(int64_t stage_id, const Shapes &inputs_shape, const Shapes &splittable_inputs,
                                          std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if (inputs_shape[0].size() >= inputs_shape[1].size()) {
    MS_LOG(ERROR) << "Invalid inputs shape.";
    return FAILED;
  }

  // first, generate strategy for input0 the same as input1
  Shapes tmp_inputs_shape = {inputs_shape[1], inputs_shape[1]};
  Shapes tmp_splittable_inputs = {splittable_inputs[1], splittable_inputs[1]};
  if (GenerateStrategiesForTwoEqualInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
    return FAILED;
  }

  // second, get the correct strategy for input0
  for (auto &sp : *sp_vector) {
    Strategies tmp_strategy;
    Dimensions input0_strategy = sp->GetInputDim()[0];
    size_t size_diff = inputs_shape[1].size() - inputs_shape[0].size();

    // erase the unnecessary part
    (void)input0_strategy.erase(input0_strategy.cbegin(),
                                input0_strategy.cbegin() + static_cast<different_type>(size_diff));

    // handle the case likes ([1, c, d], [a, b, c, d])
    for (size_t i = 0; i < inputs_shape[0].size(); ++i) {
      if (inputs_shape[0][i] == 1) {
        input0_strategy[i] = 1;
      } else {
        break;
      }
    }

    // reset the strategy
    tmp_strategy.push_back(input0_strategy);       // input0
    tmp_strategy.push_back(sp->GetInputDim()[1]);  // input1
    sp->ResetInputs(tmp_strategy);
  }
  return SUCCESS;
}

// generate strategies for that input0 and input1 have relevant dimensions, and input1 needs to broadcast
// such as: ([a, b, c, d], [b, c, d]) or ([a, b, c, d], [1, c, d])
Status GenerateStrategiesForBroadcastRight(int64_t stage_id, const Shapes &inputs_shape,
                                           const Shapes &splittable_inputs, std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if (inputs_shape[0].size() <= inputs_shape[1].size()) {
    MS_LOG(ERROR) << "Invalid inputs shape.";
    return FAILED;
  }

  // first, generate strategy for input1 the same as input0
  Shapes tmp_inputs_shape = {inputs_shape[0], inputs_shape[0]};
  Shapes tmp_splittable_inputs = {splittable_inputs[0], splittable_inputs[0]};
  if (GenerateStrategiesForTwoEqualInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
    return FAILED;
  }

  // second, get the correct strategy for input1
  for (auto &sp : *sp_vector) {
    Strategies tmp_strategy;
    tmp_strategy.push_back(sp->GetInputDim()[0]);  // input0

    Dimensions input1_strategy = sp->GetInputDim()[1];
    size_t size_diff = inputs_shape[0].size() - inputs_shape[1].size();

    // erase the unnecessary part
    (void)input1_strategy.erase(input1_strategy.cbegin(),
                                input1_strategy.cbegin() + static_cast<different_type>(size_diff));

    // handle the case likes ([a, b, c, d], [1, c, d])
    for (size_t i = 0; i < inputs_shape[1].size(); ++i) {
      if (inputs_shape[1][i] == 1) {
        input1_strategy[i] = 1;
      } else {
        break;
      }
    }

    // reset the strategy
    tmp_strategy.push_back(input1_strategy);  // input1
    sp->ResetInputs(tmp_strategy);
  }
  return SUCCESS;
}

// generate strategies for that input0 and input1 have same size, and input0 or input1 needs to broadcast
// such as: ([a, 1], [1, b]) or ([a, b, c, d], [1, b, c, d]) or ([a, b, c, 1], [1, b, c, d])
Status GenerateStrategiesForBroadcastBoth(int64_t stage_id, const Shapes &inputs_shape, const Shapes &splittable_inputs,
                                          std::vector<StrategyPtr> *const sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if (inputs_shape[0].size() != inputs_shape[1].size()) {
    MS_LOG(ERROR) << "Invalid inputs shape.";
    return FAILED;
  }

  // step1: ([a, 1], [1, b]) -> [a, b]
  Shape max_shape, splittable_vector;
  for (size_t i = 0; i < inputs_shape[0].size(); ++i) {
    if (inputs_shape[0][i] >= inputs_shape[1][i]) {
      max_shape.push_back(inputs_shape[0][i]);
      splittable_vector.push_back(splittable_inputs[0][i]);
    } else {
      max_shape.push_back(inputs_shape[1][i]);
      splittable_vector.push_back(splittable_inputs[1][i]);
    }
  }

  // step2: ([a, 1], [1, b]) -> generate strategy for ([a, b], [a, b])
  Shapes tmp_inputs_shape = {max_shape, max_shape};
  Shapes tmp_splittable_inputs = {splittable_vector, splittable_vector};
  if (GenerateStrategiesForTwoEqualInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, sp_vector) != SUCCESS) {
    MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
    return FAILED;
  }

  // step3: reset the strategy if the dimension is 1
  for (auto &sp : *sp_vector) {
    Dimensions input0_strategy = sp->GetInputDim()[0];
    Dimensions input1_strategy = sp->GetInputDim()[1];
    for (size_t i = 0; i < inputs_shape[0].size(); ++i) {
      if (inputs_shape[0][i] == 1) {
        input0_strategy[i] = 1;
      }

      if (inputs_shape[1][i] == 1) {
        input1_strategy[i] = 1;
      }
    }
    sp->ResetInputs({input0_strategy, input1_strategy});
  }

  return SUCCESS;
}

Status GenerateStrategiesForIndependentInputsBase(int64_t stage_id, size_t dev_num, const Shapes &inputs_shape,
                                                  const Shapes &splittable_inputs,
                                                  std::vector<StrategyPtr> *sp_vector) {
  Shape combined_inputs_shape, combined_splittable_inputs, combined_partitions;
  for (size_t j = 0; j < inputs_shape.size(); ++j) {
    (void)combined_inputs_shape.insert(combined_inputs_shape.cend(), inputs_shape[j].cbegin(), inputs_shape[j].cend());
    (void)combined_splittable_inputs.insert(combined_splittable_inputs.cend(), splittable_inputs[j].cbegin(),
                                            splittable_inputs[j].cend());
  }
  std::function<void(uint64_t, size_t)> recursive = [&stage_id, &dev_num, &sp_vector, &combined_inputs_shape,
                                                     &combined_splittable_inputs, &combined_partitions, &recursive,
                                                     &inputs_shape](uint64_t current_index, size_t n) {
    if (current_index == combined_inputs_shape.size()) {
      MS_LOG(DEBUG) << "The value of combined_splittable_inputs.size is: " << combined_splittable_inputs.size();
      Shapes inputs_partitions;
      size_t global_index = 0;
      for (auto &shape : inputs_shape) {
        Shape tmp_partition;
        for (size_t j = 0; j < shape.size(); ++j) {
          tmp_partition.push_back(combined_partitions[global_index]);
          global_index++;
        }
        inputs_partitions.push_back(tmp_partition);
      }
      StrategyPtr sp;
      if (PrepareStrategyBase(stage_id, dev_num, inputs_partitions, &sp) == SUCCESS) {
        sp_vector->push_back(sp);
      }
      return;
    } else {
      MS_LOG(DEBUG) << "The value of sp_vector size is " << sp_vector->size();
      if (combined_splittable_inputs[current_index] == 0) {
        combined_partitions.push_back(MIN_SLICE_NUM);
        recursive(current_index + 1, n / MIN_SLICE_NUM);
        combined_partitions.pop_back();
      } else if (combined_splittable_inputs[current_index] == 1) {
        for (uint64_t i = 1; i <= n; i *= 2) {
          if (n % i == 0 && LongToSize(combined_inputs_shape[current_index]) % i == 0) {
            combined_partitions.push_back(i);
            recursive(current_index + 1, n / i);
            combined_partitions.pop_back();
          }
        }
      }
    }
  };
  recursive(0, dev_num);
  if (sp_vector->empty()) {
    MS_LOG(EXCEPTION) << "No available strategy for current OperatorInfo.";
  }
  return SUCCESS;
}

// 'splittable_inputs' has the same dimensions as 'inputs_shape_'. '0' in 'splittable_inputs' means that
// the corresponding dimension is unsplittable, '1' in 'splittable_inputs' means that the corresponding
// dimension is splittable. 'inputs_partitions' is the result of partitions.
// NOTE: This implementation would partition all splittable dimensions in all inputs. Some operators requiring
// specific dimensions in inputs have the identical partition should have individual implementation.
Status GenerateStrategiesForIndependentInputs(int64_t stage_id, const Shapes &inputs_shape,
                                              const Shapes &splittable_inputs, std::vector<StrategyPtr> *sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }
  if (splittable_inputs.size() != inputs_shape.size()) {
    MS_LOG(ERROR) << "Splittable_inputs do not have the same input number of inputs shape, " << splittable_inputs.size()
                  << " : " << inputs_shape.size();
    return FAILED;
  }
  CheckGlobalDeviceManager();
  size_t dev_num = g_device_manager->GetDeviceListByStageId(stage_id).size();
  auto dev_num_2_power = (dev_num & (dev_num - 1));
  if (dev_num_2_power == 0) {
    return GenerateStrategiesForIndependentInputsBase(stage_id, dev_num, inputs_shape, splittable_inputs, sp_vector);
  }
  MS_EXCEPTION_IF_ZERO("dev_num - dev_num_2_power", dev_num - dev_num_2_power);
  auto dev_num_not_2_power = dev_num / (dev_num - dev_num_2_power);
  std::vector<StrategyPtr> sp_vector_2_power_part;
  if (GenerateStrategiesForIndependentInputsBase(stage_id, dev_num - dev_num_2_power, inputs_shape, splittable_inputs,
                                                 &sp_vector_2_power_part) != SUCCESS) {
    MS_LOG(ERROR) << "Generate strategy in the power of 2 devices part failed.";
    return FAILED;
  }
  // Handle the not power of 2 part.
  for (auto &stra : sp_vector_2_power_part) {
    auto stra_arrays = stra->GetInputDim();
    size_t stras_size = stra_arrays.size();
    for (size_t i = 0; i < stras_size; ++i) {
      auto split_input = splittable_inputs[i];
      size_t stra_size = stra_arrays[i].size();
      for (size_t j = 0; j < stra_size; ++j) {
        if (split_input[j] == 0) {
          continue;
        }
        auto new_stra_arrays{stra_arrays};
        new_stra_arrays[i][j] = new_stra_arrays[i][j] * UlongToLong(dev_num_not_2_power);
        // discard invalid strategy
        MS_EXCEPTION_IF_ZERO("new_stra_arrays[i][j]", new_stra_arrays[i][j]);
        if (inputs_shape[i][j] % new_stra_arrays[i][j] != 0) {
          continue;
        }
        StrategyPtr new_stra = std::make_shared<Strategy>(stage_id, new_stra_arrays);
        sp_vector->push_back(new_stra);
      }
    }
  }
  // add the repeated strategy
  auto repeated_stra_arrays{splittable_inputs};
  for (auto &stra_array : repeated_stra_arrays) {
    std::fill(stra_array.begin(), stra_array.end(), 1);
  }
  StrategyPtr repeated_stra = std::make_shared<Strategy>(stage_id, repeated_stra_arrays);
  sp_vector->push_back(repeated_stra);
  return SUCCESS;
}

// 'splittable_inputs' has the same dimensions as 'inputs_shape_'. '0' in 'splittable_inputs' means that
// the corresponding dimension is unsplittable, otherwise means that the corresponding dimension is splittable.
// In particular, if the same dimensions exist in 'splittable_inputs',
// the corresponding dimensions in the strategy are the same.
// 'sp' is the result of partitions.
Status GenerateStrategiesForDependentInputs(int64_t stage_id, const Shapes &inputs_shape,
                                            const Shapes &splittable_inputs, std::vector<StrategyPtr> *sp) {
  if (inputs_shape.size() != splittable_inputs.size()) {
    MS_LOG(EXCEPTION) << "Size of inputs_shape and splittable_inputs are not equal.";
  }

  std::unordered_map<int64_t, int64_t> mp;
  for (size_t i = 0; i < inputs_shape.size(); ++i) {
    auto input_shape = inputs_shape[i];
    auto splittable_input = splittable_inputs[i];
    for (size_t j = 0; j < input_shape.size(); ++j) {
      int64_t indice = splittable_input[j];
      int64_t shape = input_shape[j];
      if (splittable_input[j] == 0) {
        continue;
      }
      if (mp.find(indice) == mp.end()) {
        mp[indice] = shape;
      } else {
        mp[indice] = std::gcd(mp[indice], shape);
      }
    }
  }

  std::unordered_map<int64_t, size_t> indices_mp;
  Shape tmp_input_shape;
  Shapes tmp_splittable_inputs = {Shape(mp.size(), 1)};

  for (const auto &item : mp) {
    indices_mp[item.first] = tmp_input_shape.size();
    tmp_input_shape.push_back(item.second);
  }
  Shapes tmp_inputs_shape = {tmp_input_shape};
  std::vector<StrategyPtr> tmp_sp_vector;
  if (GenerateStrategiesForIndependentInputs(stage_id, tmp_inputs_shape, tmp_splittable_inputs, &tmp_sp_vector) !=
      SUCCESS) {
    return FAILED;
  }

  (void)std::transform(tmp_sp_vector.begin(), tmp_sp_vector.end(), std::back_inserter(*sp),
                       [stage_id, &indices_mp, &splittable_inputs](const StrategyPtr &sp) {
                         auto sp_strategies = sp->GetInputDim();
                         auto sp_sub_strategy = sp_strategies.at(0);
                         Strategies strategies(splittable_inputs);
                         for (size_t i = 0; i < strategies.size(); ++i) {
                           for (size_t j = 0; j < strategies[i].size(); ++j) {
                             if (splittable_inputs[i][j] == 0) {
                               strategies[i][j] = 1;
                             } else {
                               strategies[i][j] = sp_sub_strategy[indices_mp[splittable_inputs[i][j]]];
                             }
                           }
                         }
                         return std::make_shared<Strategy>(stage_id, strategies);
                       });
  return SUCCESS;
}

// generate strategies for that have two inputs, and input0 or input1 maybe broadcast,
// and the corresponding dimensions that are not broadcast are all relevant dimensions
// such as: ([a, b, c, d], [a, b, c, d]) or ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
// or ([a, b, c, d], [b, c, d]) or ([a, b, c, d], [1, c, d])
// or ([a, 1], [1, b]) or ([a, b, c, d], [1, b, c, d]) or ([a, b, c, 1], [1, b, c, d])
Status GenerateStrategiesWithBroadcast(int64_t stage_id, const Shapes &inputs_shape, const Shapes &splittable_inputs,
                                       std::vector<StrategyPtr> *sp_vector) {
  if (sp_vector == nullptr) {
    MS_LOG(ERROR) << "The sp_vector is null.";
    return FAILED;
  }

  if ((inputs_shape.size() != 2) || (splittable_inputs.size() != 2)) {
    MS_LOG(ERROR) << "The inputs' size is wrong.";
    return FAILED;
  }

  if (inputs_shape[0] == inputs_shape[1]) {
    // element wise operation([a, b, c, d], [a, b, c, d]), so input0's strategy is equal to input1's strategy
    if (GenerateStrategiesForTwoEqualInputs(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForTwoEqualInputs failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForTwoEqualInputs success.";
  } else if (inputs_shape[0].empty() || inputs_shape[1].empty()) {
    // ([a, b, c, d], []) or ([], [a, b, c, d])
    if (GenerateStrategiesForIndependentInputs(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "Generate strategies for scalar case failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "Generate strategies for scalar case success.";
  } else if (inputs_shape[0].size() > inputs_shape[1].size()) {
    // ([a, b, c, d], [b, c, d]) or ([a, b, c, d], [1, c, d])
    if (GenerateStrategiesForBroadcastRight(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForBroadcastRight failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForBroadcastRight success.";
  } else if (inputs_shape[0].size() < inputs_shape[1].size()) {
    // ([b, c, d], [a, b, c, d]) or ([1, c, d], [a, b, c, d])
    if (GenerateStrategiesForBroadcastLeft(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForBroadcastLeft failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForBroadcastLeft success.";
  } else {  // same size, but different value
    // ([a, 1], [1, b]) or ([a, b, c, d], [1, b, c, d]) or ([a, b, c, 1], [1, b, c, d])
    if (GenerateStrategiesForBroadcastBoth(stage_id, inputs_shape, splittable_inputs, sp_vector) != SUCCESS) {
      MS_LOG(ERROR) << "GenerateStrategiesForBroadcastBoth failed.";
      return FAILED;
    }
    MS_LOG(INFO) << "GenerateStrategiesForBroadcastBoth success.";
  }
  return SUCCESS;
}

CostPtr OperatorInfo::ComputeCost(const StrategyPtr &strategy) {
  if (strategy == nullptr) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << "ComputeCost failed: strategy is null.";
  }
  int64_t stage_id = strategy->GetInputStage();
  double computation_cost =
    operator_cost()->GetForwardComputationCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  double communication_cost = operator_cost()->GetCommCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  const auto gamma = CostModelContext::GetInstance()->costmodel_gamma();
  CostPtr result = std::make_shared<Cost>(computation_cost, communication_cost);
  result->communication_without_parameter_ =
    operator_cost()->GetForwardCommCost(inputs_tensor_info_, outputs_tensor_info_, stage_id);
  result->communication_with_partial_para_ =
    result->communication_without_parameter_ + gamma * (communication_cost - result->communication_without_parameter_);

  // Breaking ties for preferring data parallelization
  BreakingTiesForPreferringDataParallel(strategy, result);
  // refine communication cost calculation for practice
  RefineForPracticalCost(result, false);
  result->communication_forward_ = result->communication_without_parameter_;
  return result;
}

Status OperatorInfo::SetCostUnderStrategyBase(const StrategyPtr &strategy) {
  StrategyPtr out_strategy = out_strategy_;
  if (InitForCostModel(strategy, out_strategy) == FAILED) {
    MS_LOG(DEBUG) << name_ << ": Initialization under the strategy failed.";
    return FAILED;
  }

  CostPtr result = ComputeCost(strategy);
  std::shared_ptr<StrategyWithCost> swc =
    std::make_shared<StrategyWithCost>(strategy, inputs_tensor_info_, outputs_tensor_info_);
  swc->cost_list.push_back(result);
  (void)strategy_cost_.emplace_back(swc);

  return SUCCESS;
}

void OperatorInfo::SetDefaultLayoutInfo() {
  if (SetDevMatrixShapeByLayout() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": SetDevMatrixShapeByLayout failed.";
  }
  if (SetTensorMapByLayout() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": SetTensorMapByLayout failed.";
  }
  if (SetTensorMapBeforeByLayout() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": SetTensorMapBeforeByLayout failed.";
  }
  if (SetOutDevMatrixShapeByLayout() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": SetOutDevMatrixShapeByLayout failed.";
  }
  if (SetOutTensorMapByLayout() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": SetOutTensorMapByLayout failed.";
  }
  if (SetOutTensorMapBeforeByLayout() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": SetOutTensorMapBeforeByLayout failed.";
  }
}

Status OperatorInfo::SetCostUnderLayout(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy,
                                        const std::vector<std::shared_ptr<TensorLayout>> &in_tensor_layouts,
                                        const std::vector<std::shared_ptr<TensorLayout>> &out_tensor_layouts) {
  if (Init(in_strategy, out_strategy, in_tensor_layouts, out_tensor_layouts) == FAILED) {
    MS_LOG(DEBUG) << name_ << ": Initialization under the layout failed.";
    return FAILED;
  }

  strategy_ = in_strategy;
  out_strategy_ = out_strategy;

  CostPtr result = ComputeCost(in_strategy);
  std::shared_ptr<StrategyWithCost> swc =
    std::make_shared<StrategyWithCost>(in_strategy, inputs_tensor_info_, outputs_tensor_info_);
  swc->cost_list.push_back(result);
  (void)strategy_cost_.emplace_back(swc);

  SetDefaultLayoutInfo();

  return SUCCESS;
}

Status OperatorInfo::SetCostUnderStrategyWithCost(const std::shared_ptr<StrategyWithCost> &swc) {
  StrategyPtr out_strategy = out_strategy_;
  ResetQueueMember();

  if (InferAttrs() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": InferAttrs failed.";
    return FAILED;
  }

  strategy_ = swc->strategy_ptr;
  out_strategy_ = out_strategy;
  inputs_tensor_info_ = swc->inputs_ptr;
  outputs_tensor_info_ = swc->outputs_ptr;
  strategy_cost_.push_back(swc);

  if (CheckInputLayout() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": CheckInputLayout failed.";
    return FAILED;
  }

  if (outputs_tensor_info_.size() != outputs_shape_.size()) {
    outputs_tensor_info_.clear();
    // Need be override
    if (InferOutputTensorInfo() != SUCCESS) {
      MS_LOG(WARNING) << name_ << ": InferOutputTensorLayout failed.";
      return FAILED;
    }
  }

  if (outputs_tensor_info_.size() != outputs_shape_.size()) {
    MS_LOG(WARNING) << name_ << ": the output tensor layout num " << outputs_tensor_info_.size()
                    << " does not match the output num " << outputs_shape_.size();
    return FAILED;
  }

  SetDefaultLayoutInfo();

  if (CheckOutputLayout() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": CheckLayout failed.";
    return FAILED;
  }
  // Need be override
  if (InferForwardCommunicationByLayout() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": InferForwardCommunication failed.";
    return FAILED;
  }

  if (InferMirrorOpsByLayout() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": InferMirrorOps failed.";
    return FAILED;
  }
  if (InferVirtualDivOpsByLayout() != SUCCESS) {
    MS_LOG(WARNING) << name_ << ": InferVirtualDivOps failed.";
    return FAILED;
  }
  InferReplaceOps();
  return SUCCESS;
}

Status OperatorInfo::SetDevMatrixShapeByLayout() {
  dev_matrix_shape_.clear();
  for (const auto &tensor_info : inputs_tensor_info_) {
    TensorLayout layout = tensor_info.tensor_layout();
    Arrangement device_arrangement = layout.device_arrangement_origin();
    if (!dev_matrix_shape_.empty() && dev_matrix_shape_ != device_arrangement.array()) {
      MS_LOG(INFO) << "Not support different device matrix now.";
      return FAILED;
    }
    dev_matrix_shape_ = device_arrangement.array();
  }
  return SUCCESS;
}

Status OperatorInfo::SetTensorMapByLayout() {
  inputs_tensor_map_.clear();
  (void)std::transform(
    inputs_tensor_info_.begin(), inputs_tensor_info_.end(), std::back_inserter(inputs_tensor_map_),
    [&](const TensorInfo &tensor_info) { return tensor_info.tensor_layout().origin_tensor_map().array(); });
  return SUCCESS;
}

Status OperatorInfo::SetTensorMapBeforeByLayout() {
  inputs_tensor_map_before_.clear();
  (void)std::transform(inputs_tensor_info_.begin(), inputs_tensor_info_.end(),
                       std::back_inserter(inputs_tensor_map_before_),
                       [&](const TensorInfo &tensor_info) { return tensor_info.tensor_layout().tensor_map_before(); });
  return SUCCESS;
}

Status OperatorInfo::SetOutDevMatrixShapeByLayout() {
  out_dev_matrix_shape_.clear();
  for (const auto &tensor_info : outputs_tensor_info_) {
    TensorLayout layout = tensor_info.tensor_layout();
    Arrangement device_arrangement = layout.device_arrangement_origin();
    if (!out_dev_matrix_shape_.empty() && out_dev_matrix_shape_ != device_arrangement.array()) {
      MS_LOG(ERROR) << "Not support different device matrix now.";
      return FAILED;
    }
    out_dev_matrix_shape_ = device_arrangement.array();
  }
  return SUCCESS;
}

Status OperatorInfo::SetOutTensorMapByLayout() {
  outputs_tensor_map_.clear();
  (void)std::transform(
    outputs_tensor_info_.begin(), outputs_tensor_info_.end(), std::back_inserter(outputs_tensor_map_),
    [&](const TensorInfo &tensor_info) { return tensor_info.tensor_layout().origin_tensor_map().array(); });
  return SUCCESS;
}

Status OperatorInfo::SetOutTensorMapBeforeByLayout() {
  outputs_tensor_map_before_.clear();
  (void)std::transform(outputs_tensor_info_.begin(), outputs_tensor_info_.end(),
                       std::back_inserter(outputs_tensor_map_before_),
                       [&](const TensorInfo &tensor_info) { return tensor_info.tensor_layout().tensor_map_before(); });
  return SUCCESS;
}

TensorLayout OperatorInfo::GetInputLayoutFromSWCByStrategy(const StrategyPtr &stra, size_t input_index) {
  auto is_target = [&](const std::shared_ptr<StrategyWithCost> &swc) { return swc->strategy_ptr->IsEqual(stra); };
  auto it = std::find_if(strategy_cost_.begin(), strategy_cost_.end(), is_target);
  if (it != strategy_cost_.end()) {
    const auto &input_info = (*it)->inputs_ptr[input_index];
    return std::move(input_info.tensor_layout());
  }
  TensorLayout empty;
  return empty;
}

TensorLayout OperatorInfo::GetOutputLayoutFromSWCByStrategy(const StrategyPtr &stra, size_t output_index) {
  auto is_target = [&](const std::shared_ptr<StrategyWithCost> &swc) { return swc->strategy_ptr->IsEqual(stra); };
  auto it = std::find_if(strategy_cost_.begin(), strategy_cost_.end(), is_target);
  if (it != strategy_cost_.end()) {
    const auto &output_info = (*it)->outputs_ptr[output_index];
    return std::move(output_info.tensor_layout());
  }
  TensorLayout empty;
  return empty;
}

StrategyPtr OperatorInfo::GetStrategyFromSWCByInputLayout(const TensorLayout &input_layout, size_t input_index) {
  auto is_target = [&](const std::shared_ptr<StrategyWithCost> &swc) {
    return swc->inputs_ptr[input_index].tensor_layout() == input_layout;
  };
  auto it = std::find_if(strategy_cost_.begin(), strategy_cost_.end(), is_target);
  if (it != strategy_cost_.end()) {
    return (*it)->strategy_ptr;
  }
  return nullptr;
}

StrategyPtr OperatorInfo::GetStrategyFromSWCByOutputLayout(const TensorLayout &output_layout, size_t output_index) {
  auto is_target = [&](const std::shared_ptr<StrategyWithCost> &swc) {
    return swc->outputs_ptr[output_index].tensor_layout() == output_layout;
  };
  auto it = std::find_if(strategy_cost_.begin(), strategy_cost_.end(), is_target);
  if (it != strategy_cost_.end()) {
    return (*it)->strategy_ptr;
  }
  return nullptr;
}

std::vector<std::shared_ptr<StrategyWithCost>> OperatorInfo::GetSwcByInputLayout(const TensorLayout &input_layout,
                                                                                 size_t input_index) {
  std::vector<std::shared_ptr<StrategyWithCost>> matchedSwcs;
  auto is_target = [&](const std::shared_ptr<StrategyWithCost> &swc) {
    return swc->inputs_ptr[input_index].tensor_layout() == input_layout;
  };
  std::copy_if(strategy_cost_.begin(), strategy_cost_.end(), std::back_inserter(matchedSwcs), is_target);
  return matchedSwcs;
}

std::vector<std::shared_ptr<StrategyWithCost>> OperatorInfo::GetSwcByOutputLayout(const TensorLayout &output_layout,
                                                                                  size_t output_index) {
  std::vector<std::shared_ptr<StrategyWithCost>> matchedSwcs;
  auto is_target = [&](const std::shared_ptr<StrategyWithCost> &swc) {
    return swc->outputs_ptr[output_index].tensor_layout() == output_layout;
  };
  std::copy_if(strategy_cost_.begin(), strategy_cost_.end(), std::back_inserter(matchedSwcs), is_target);
  return matchedSwcs;
}

bool OperatorInfo::IsReshape() const {
  if (name_.find(RESHAPEINFO) != std::string::npos) {
    return true;
  }
  return false;
}

bool OperatorInfo::IsVirtualOutput() const {
  if (name_.find(VIRTUALOUTPUTINFO) != std::string::npos) {
    return true;
  }
  return false;
}

bool OperatorInfo::IsConcat() const {
  if (name_.find(CONCATINFO) != std::string::npos) {
    return true;
  }
  return false;
}

bool OperatorInfo::IsStandAlone() const {
  if (name_.find(STAND_ALONE_INFO) != std::string::npos) {
    return true;
  }
  return false;
}

bool OperatorInfo::IsTmpIdentity() const {
  if (name_.find(IDENTITY_INFO) != std::string::npos) {
    return true;
  }
  return false;
}

bool OperatorInfo::IsMultiInput() const {
  MS_LOG(INFO) << "OperatorInfo::IsMultiInput inputs_shape_.size(): " << inputs_shape_.size();
  if (inputs_shape_.size() < INT64_TWO) {
    return false;
  }
  MS_LOG(INFO) << "OperatorInfo::IsMultiInput inputs[0]: " << inputs_shape_[INDEX_ZERO]
               << "inputs[1]: " << inputs_shape_[INDEX_ONE];
  return true;
}

bool CompareSwc(const std::pair<std::shared_ptr<StrategyWithCost>, std::pair<double, double>> &a,
                const std::pair<std::shared_ptr<StrategyWithCost>, std::pair<double, double>> &b) {
  if (!common::IsDoubleEqual(a.second.first, b.second.first)) {
    return a.second.first < b.second.first;
  }

  if (!common::IsDoubleEqual(a.second.second, b.second.second)) {
    return a.second.second < b.second.second;
  }

  if (!common::IsDoubleEqual(a.first->cost_list[0]->computation_cost_, b.first->cost_list[0]->computation_cost_)) {
    return a.first->cost_list[0]->computation_cost_ < b.first->cost_list[0]->computation_cost_;
  }
  return a.first->cost_list[0]->communication_without_parameter_ <
         b.first->cost_list[0]->communication_without_parameter_;
}

bool OperatorInfo::AllInputsVisited() const {
  MS_LOG(INFO) << "op: " << name_ << " visited_edges_.size(): " << visited_edges_.size()
               << " inputs_shape_.size(): " << inputs_shape_.size();
  return visited_edges_.size() == inputs_shape_.size();
}

std::shared_ptr<StrategyWithCost> OperatorInfo::GetStrategyByVisitedEdges() {
  MS_LOG(INFO) << "op: " << name_ << " GetStrategyByVisitedEdges";
  std::vector<std::pair<std::shared_ptr<StrategyWithCost>, std::pair<double, double>>> cur_op_swcs;
  for (std::shared_ptr<StrategyWithCost> &cur_op_swc : strategy_cost_) {
    MS_LOG(INFO) << "cur_op_swc strategy: " << cur_op_swc->strategy_ptr->ToString();
    MS_LOG(INFO) << "cur_op_swc strategy computation cost: " << cur_op_swc->cost_list[0]->computation_cost_;
    double communication_cost_sum = 0.0;
    double computation_cost_sum = 0.0;
    bool strategy_valid = true;
    for (auto &visited_edge : visited_edges_) {
      size_t visited_op_output_index = visited_edge->prev_op_output_index();
      const TensorLayout &visited_op_layout =
        visited_edge->prev_operator()->outputs_tensor_info()[visited_op_output_index].tensor_layout();

      size_t cur_op_input_index = visited_edge->next_op_input_index();
      const TensorLayout &cur_op_layout = cur_op_swc->inputs_ptr[cur_op_input_index].tensor_layout();

      CostPtrKey ck = {visited_op_layout, cur_op_layout};

      const CostPtr &cost = visited_edge->GetCostByLayoutPair(ck);

      MS_LOG(INFO) << "visited_op: " << visited_edge->prev_operator()->name()
                   << " layout: " << visited_op_layout.ToString();
      MS_LOG(INFO) << "cur op layout: " << cur_op_layout.ToString();
      if (cost == nullptr) {
        MS_LOG(WARNING) << "cost nullptr";
        strategy_valid = false;
        break;
      }
      MS_LOG(INFO) << "communication cost: " << cost->communication_cost_
                   << ", computation cost:" << cost->computation_cost_;
      communication_cost_sum += cost->communication_cost_;
      computation_cost_sum += cost->computation_cost_;
    }
    if (!strategy_valid) {
      MS_LOG(WARNING) << "strategy invalid, continue searching";
      continue;
    }
    MS_LOG(INFO) << "communication cost sum: " << communication_cost_sum
                 << ", computation cost:" << computation_cost_sum;
    cur_op_swcs.emplace_back(cur_op_swc, std::make_pair(communication_cost_sum, computation_cost_sum));
  }
  auto min_cur_op_swc = std::min_element(cur_op_swcs.begin(), cur_op_swcs.end(), CompareSwc);
  MS_LOG(INFO) << "Return cur strategy selected by visited edges: " << min_cur_op_swc->first->strategy_ptr->ToString();
  return min_cur_op_swc->first;
}

// Keep at most (1.0 / epsilon) number of available strategies for each operator.
void OperatorInfo::ApproximateStrategies() {
  auto enable_approxi = CostModelContext::GetInstance()->dp_algo_enable_approxi();
  if (!enable_approxi) {
    return;
  }
  MS_LOG(INFO) << name_ << ": Approximating strategy-cost";
  auto epsilon = CostModelContext::GetInstance()->dp_algo_approxi_epsilon();
  MS_EXCEPTION_IF_ZERO("epsilon", epsilon);
  auto target_num = static_cast<size_t>(std::ceil(1.0 / epsilon));
  if (strategy_cost_.size() <= target_num) {
    MS_LOG(INFO) << name_ << "'s strategy number is: " << strategy_cost_.size()
                 << ", no greater than target-num: " << target_num;
    return;
  }
  std::vector<std::shared_ptr<StrategyWithCost>> ret;
  auto &origin_stra_cost = strategy_cost_;
  auto alpha = CostModelContext::GetInstance()->costmodel_alpha();
  auto beta = CostModelContext::GetInstance()->costmodel_beta();
  // sort
  std::sort(
    origin_stra_cost.begin(), origin_stra_cost.end(),
    [&alpha, &beta](const std::shared_ptr<StrategyWithCost> &s1, const std::shared_ptr<StrategyWithCost> &s2) {
      if (alpha * s1->cost_list[0]->computation_cost_ + beta * s1->cost_list[0]->communication_with_partial_para_ <
          alpha * s2->cost_list[0]->computation_cost_ + beta * s2->cost_list[0]->communication_with_partial_para_) {
        return true;
      }
      return false;
    });
  MS_EXCEPTION_IF_ZERO("target_num", target_num);
  size_t step_length = origin_stra_cost.size() / target_num;
  for (size_t i = 0; ret.size() < target_num && static_cast<size_t>(i * step_length) < origin_stra_cost.size(); ++i) {
    ret.push_back(origin_stra_cost[static_cast<size_t>(i * step_length)]);
  }

  strategy_cost_ = ret;
  is_strategy_cost_exact_ = false;
}

void OperatorInfo::ExactStrategiesAndRelatedEdges() {
  if (is_strategy_cost_exact()) {
    return;
  }
  ClearStrategyCost();
  if (GenerateStrategies(0) != SUCCESS) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Strategy search failed.";
  }
  SetIsStrategyCostExactTrue();
  // re-init the previous edges
  for (auto &prev_edge : prev_edges()) {
    if (prev_edge->InitEdgeCost() != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Edge: " << prev_edge->edge_name() << " cost init failed.";
    }
  }
  // re-init the successive edges
  for (auto &next_edge : succ_edges()) {
    if (next_edge->InitEdgeCost() != SUCCESS) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Edge: " << next_edge->edge_name() << " cost init failed.";
    }
  }
}

int64_t OperatorInfo::ComputeOpAndPrevEdgeParameterInvolved() {
  if (is_output_parameter_involve_ != -1) {
    return is_output_parameter_involve_;
  }
  is_parameter_involve_ = is_parameter_;
  const auto &prev_edges = this->GetAlivePrevEdges();
  for (auto &p_edge : prev_edges) {
    auto input_index = p_edge->next_op_input_index();
    auto prev_op_para = p_edge->prev_operator()->ComputeOpAndPrevEdgeParameterInvolved();
    if (input_index >= is_parameter_involve_.size()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << " has input length: " << is_parameter_involve_.size()
                                          << ", but got wrong input_index: " << input_index;
    }
    if (prev_op_para == 0) {
      is_parameter_involve_[input_index] = false;
    } else if (prev_op_para == 1) {
      is_parameter_involve_[input_index] = true;
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_)
        << name_ << " got wrong value: " << prev_op_para << ", input_index: " << input_index;
    }
    p_edge->set_parameter_involve(prev_op_para);
  }
  if (std::any_of(is_parameter_involve_.begin(), is_parameter_involve_.end(), [](bool value) { return value; })) {
    // If anyone of the input is a parameter_involved, the output is parameter_involved.
    is_output_parameter_involve_ = 1;
  } else {
    is_output_parameter_involve_ = 0;
  }
  // Set 'is_parameter_involve_' and 'is_output_parameter_involve_' into operatorCost, which are used in
  // calculating 'inputs_in_memory' and 'output_in_memory', respectively.
  operator_cost()->set_is_parameter_involve(is_parameter_involve_);
  operator_cost()->set_output_parameter_involve(is_output_parameter_involve_);
  // Calculating 'output_in_memory'
  operator_cost()->CalculateOutputInMemory();
  // Calculating 'inputs_in_memory'
  std::map<size_t, bool> input_in_memory;
  for (auto &p_edge : prev_edges) {
    auto input_index = p_edge->next_op_input_index();
    auto is_in_mem = p_edge->prev_operator()->operator_cost()->is_output_in_memory();
    (void)input_in_memory.emplace(std::make_pair(input_index, is_in_mem));
  }
  operator_cost()->CalculateInputsInMemory(input_in_memory);

  return is_output_parameter_involve_;
}

Status OperatorInfo::set_is_parameter(const std::vector<bool> &is_parameter) {
  if (inputs_shape_new_.size() == 0) {
    if (is_parameter.size() != inputs_shape_.size()) {
      MS_LOG(ERROR) << name_ << ": Is_parameter: " << is_parameter.size()
                    << " do not have the same number of inputs_shape_: " << inputs_shape_.size();
      return FAILED;
    }
  }
  is_parameter_ = is_parameter;
  operator_cost()->set_is_parameter(is_parameter);
  return SUCCESS;
}

Status OperatorInfo::CalculateMemoryCost() {
  if (is_parameter_involve_.size() != is_parameter_.size()) {
    MS_LOG(ERROR) << name_ << ": the size of 'is_parameter_' is " << is_parameter_.size()
                  << " does not have the same number of the size of 'is_parameter_involve_'."
                  << is_parameter_involve_.size();
    return FAILED;
  }
  // Set the memory cost in the 'strategy_cost_'
  for (auto &swc : strategy_cost_) {
    auto mem_cost = operator_cost()->GetMemoryCost(swc->inputs_ptr, swc->outputs_ptr);
    swc->cost_list[0]->memory_with_reuse_ = mem_cost;
  }
  return SUCCESS;
}

Status OperatorInfo::CalculateMemoryCostForInference() {
  // First, set the 'is_outputs_critical_' flag into OperatorCost.
  if (is_output_critical_ == -1) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": The critical flag is not set.";
  }
  operator_cost()->set_output_critical(is_output_critical_);
  // Set the memory cost in the 'strategy_cost_'
  for (auto &swc : strategy_cost_) {
    auto mem_cost = operator_cost()->GetMemoryCostForInference(swc->inputs_ptr, swc->outputs_ptr);
    swc->cost_list[0]->memory_with_reuse_ = mem_cost;
  }
  return SUCCESS;
}

Status OperatorInfo::CorrectMemoryCost(size_t input_index) {
  for (auto &swc : strategy_cost_) {
    double parameter_mem_cost = ListProduct(swc->inputs_ptr[input_index].slice_shape()) *
                                static_cast<double>(operator_cost()->inputs_type_lengths()[input_index]);
    swc->cost_list[0]->memory_with_reuse_ -= parameter_mem_cost;
    if (swc->cost_list[0]->memory_with_reuse_ < 0) {
      MS_LOG(WARNING) << name_ << ": The memory cost after correction is: " << swc->cost_list[0]->memory_with_reuse_
                      << ", the parameter memory cost is: " << parameter_mem_cost;
      swc->cost_list[0]->memory_with_reuse_ = 0;
    }
  }
  return SUCCESS;
}

int64_t ComputeRepeatDeviceNumByTensorMap(const Shape &dev_matrix_shape, const Shape &tensor_map) {
  int64_t ret = -1;

  // The number of repetitions is equal to the number of all devices divided by the number of devices use for
  // tensor map.
  int64_t device_num = std::accumulate(dev_matrix_shape.begin(), dev_matrix_shape.end(), 1, std::multiplies<int64_t>());
  for (auto &element : tensor_map) {
    // -1 means the corresponding dimension is not split.
    if (element == MAP_NONE) {
      continue;
    } else if ((element < 0) || (LongToSize(element) >= dev_matrix_shape.size())) {
      MS_LOG(ERROR) << "Invalid tensor map: " << ShapeToString(tensor_map) << ", the dev matrix shape is "
                    << ShapeToString(dev_matrix_shape);
      return ret;
    } else {
      size_t index = dev_matrix_shape.size() - LongToSize(element) - 1;
      if (dev_matrix_shape[index] <= 0) {
        MS_LOG(ERROR) << "Invalid dev matrix shape: " << ShapeToString(dev_matrix_shape);
        return ret;
      }
      device_num /= dev_matrix_shape[index];
    }
  }

  return device_num;
}

Status OperatorInfo::InferAsLossDivisor() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }
  if (!inputs_shape_new_.empty()) {
    MS_LOG(ERROR) << name_ << ": For Tuple input ops, please override this function";
    return FAILED;
  }
  if (outputs_tensor_map_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor map is empty.";
    return FAILED;
  }

  if (outputs_tensor_map_.size() > 1) {
    MS_LOG(ERROR) << name_ << ": The output size is " << outputs_tensor_map_.size()
                  << ", need to override this function ";
    return FAILED;
  }

  if (outputs_tensor_map_[0].empty()) {
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  if (out_dev_matrix_shape_.empty()) {
    out_dev_matrix_shape_ = dev_matrix_shape_;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(out_dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(out_dev_matrix_shape_)
               << ", the output tensor map is " << ShapeToString(outputs_tensor_map_[0]) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

Status OperatorInfo::InferAsLossDivisorByLayout() {
  if (!ParallelContext::GetInstance()->loss_repeated_mean()) {
    as_loss_divisor_ = 1;
    return SUCCESS;
  }

  if (outputs_tensor_info_.empty()) {
    MS_LOG(ERROR) << name_ << ": The outputs tensor info is empty.";
    return FAILED;
  }

  if (outputs_tensor_info_.size() > 1) {
    MS_LOG(ERROR) << name_ << ": The output size is " << outputs_tensor_info_.size()
                  << ", need to override this function ";
    return FAILED;
  }

  TensorMaps outputs_tensor_map = outputs_tensor_info_[0].tensor_layout().tensor_map_before();
  if (outputs_tensor_map.empty()) {
    MS_LOG(INFO) << name_ << ": out_dev_matrix_shape is empty";
    as_loss_divisor_ = stage_device_size_;
    MS_LOG(INFO) << name_ << ": The output is a scalar, use the dev size " << as_loss_divisor_ << ", loss divisor.";
    return SUCCESS;
  }

  auto out_dev_matrix_shape = outputs_tensor_info_[0].tensor_layout().device_arrangement_origin().array();
  if (out_dev_matrix_shape.empty()) {
    out_dev_matrix_shape = dev_matrix_shape_;
  }
  Shape squashed_tensor_map;
  for (const auto &tensor_map : outputs_tensor_map) {
    std::copy(tensor_map.begin(), tensor_map.end(), std::back_inserter(squashed_tensor_map));
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(out_dev_matrix_shape, squashed_tensor_map);
  MS_LOG(INFO) << name_ << ": the dev matrix shape is " << ShapeToString(out_dev_matrix_shape)
               << ", the output tensor map is " << ShapeToString(squashed_tensor_map) << ", loss divisor is "
               << as_loss_divisor_;
  return SUCCESS;
}

// If the operator is used as a loss, a div node is inserted for the grad of all its inputs.
Status OperatorInfo::InferVirtualDivOps() {
  if (InferAsLossDivisor() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferAsLossDivisor failed.";
    return FAILED;
  }

  if (as_loss_divisor_ <= 0) {
    MS_LOG(ERROR) << name_ << ": Invalid loss divisor: " << as_loss_divisor_;
    return FAILED;
  } else if (as_loss_divisor_ == 1) {
    MS_LOG(INFO) << name_ << ": The loss divisor is 1, no need to create virtual div op.";
    return SUCCESS;
  }

  virtual_div_op_.clear();
  // if loss is repeated calculation, insert div op
  Operator op = CreateVirtualDivOp(as_loss_divisor_);
  virtual_div_op_.push_back(op);
  return SUCCESS;
}

Status OperatorInfo::InferVirtualDivOpsByLayout() {
  if (InferAsLossDivisorByLayout() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": InferAsLossDivisor failed.";
    return FAILED;
  }

  if (as_loss_divisor_ <= 0) {
    MS_LOG(ERROR) << name_ << ": Invalid loss divisor: " << as_loss_divisor_;
    return FAILED;
  } else if (as_loss_divisor_ == 1) {
    MS_LOG(INFO) << name_ << ": The loss divisor is 1, no need to create virtual div op.";
    return SUCCESS;
  }

  virtual_div_op_.clear();
  // if loss is repeated calculation, insert div op
  Operator op = CreateVirtualDivOp(as_loss_divisor_);
  virtual_div_op_.push_back(op);
  return SUCCESS;
}

Status OperatorInfo::SetInputAndOutputTypeLength(const std::vector<size_t> &input_lengths,
                                                 const std::vector<size_t> &output_lengths) {
  if (inputs_shape_new_.size() == 0) {
    if (input_lengths.size() != inputs_shape_.size()) {
      MS_LOG(ERROR) << name_ << ": Input_lengths: " << input_lengths.size()
                    << " do not have the same number of inputs shape: " << inputs_shape_.size();
      return FAILED;
    }
  }
  if (outputs_shape_new_.size() == 0) {
    if (output_lengths.size() != outputs_shape_.size()) {
      MS_LOG(ERROR) << name_ << ": Output_lengths: " << output_lengths.size()
                    << " do not have the same number of outputs shape: " << outputs_shape_.size();
      return FAILED;
    }
  }
  inputs_type_lengths_ = input_lengths;
  outputs_type_lengths_ = output_lengths;
  operator_cost()->SetInputAndOutputTypeLength(input_lengths, output_lengths);
  return SUCCESS;
}

double OperatorInfo::GetOutputsTotalSize() {
  if (is_calculated_outputs_size_) {
    return outputs_total_size_;
  }
  if (outputs_type_lengths_.size() != outputs_shape_.size()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Output_lengths: " << outputs_type_lengths_.size()
                                        << " do not have the same number of outputs shape: " << outputs_shape_.size();
  }
  double sum = 0.0;
  for (size_t i = 0; i < outputs_type_lengths_.size(); ++i) {
    auto size = std::accumulate(outputs_shape_[i].begin(), outputs_shape_[i].end(), static_cast<double>(1.0),
                                std::multiplies<double>());
    sum += size * static_cast<double>(outputs_type_lengths_[i]);
  }
  is_calculated_outputs_size_ = true;
  outputs_total_size_ = sum;
  return outputs_total_size_;
}

Status OperatorInfo::set_outputs_type(const std::vector<TypePtr> &outputs_type) {
  if (outputs_shape_new_.size() == 0) {
    if (outputs_type.size() != outputs_shape_.size()) {
      MS_LOG(ERROR) << name_ << ": Outputs type: " << outputs_type.size()
                    << " do not have the same number of outputs shape: " << outputs_shape_.size();
      return FAILED;
    }
  }
  outputs_type_ = outputs_type;
  return SUCCESS;
}

void OperatorInfo::BreakingTiesForPreferringDataParallel(const StrategyPtr &stra, const CostPtr &cost) const {
  if (!stra->GetInputDim().empty() && !stra->GetInputDim()[0].empty()) {
    if (stra->GetInputDim()[0][0] == stage_device_size_) {
      if (cost->computation_cost_ > 1.0) {
        cost->computation_cost_ -= 1.0;
      }
      if (cost->communication_cost_ > 1.0) {
        cost->communication_cost_ -= 1.0;
      }
      if (cost->communication_with_partial_para_ > 1.0) {
        cost->communication_with_partial_para_ -= 1.0;
      }
      if (cost->communication_without_parameter_ > 1.0) {
        cost->communication_without_parameter_ -= 1.0;
      }
    }
  }
}

void OperatorInfo::SetSelectedStrategy(const StrategyPtr &s_strategy, size_t curr_depth) {
  MS_EXCEPTION_IF_NULL(s_strategy);
  if ((selected_strategy_depth_ != -1) && (SizeToLong(curr_depth) > selected_strategy_depth_)) {
    MS_LOG(INFO) << name_ << " has already been set strategy.";
    return;
  }
  MS_LOG(INFO) << name_ << ": Set strategy " << s_strategy->ToString();
  selected_strategy_ = s_strategy;
  selected_strategy_depth_ = SizeToLong(curr_depth);
}

void OperatorInfo::set_swc_index(int64_t swc, int64_t depth) {
  MS_LOG(INFO) << name_ << ": Set SWC index: " << swc;
  selected_strategy_depth_ = depth;
  swc_index_ = swc;
}

std::vector<CNodePtr> OperatorInfo::cnodes() { return cnodes_; }

double OperatorInfo::GetForwardMemoryCostFromCNode() {
  return operator_cost()->GetForwardComputationCost(inputs_tensor_info_, outputs_tensor_info_, 0);
}

void OperatorInfo::CheckSelectedStrategy(const StrategyPtr &s_strategy) {
  MS_EXCEPTION_IF_NULL(s_strategy);
  if (!s_strategy->IsEqual(selected_strategy_)) {
    MS_LOG(INFO) << name_
                 << "'s strategy may cause suboptimal, the determined strategy: " << selected_strategy_->ToString()
                 << "The minimal strategy: " << s_strategy->ToString();
  }
}

void OperatorInfo::SetStrategyCost(const std::vector<std::shared_ptr<StrategyWithCost>> &stra_cost) {
  strategy_cost_ = stra_cost;
}

Status OperatorInfo::GenerateStrategies(int64_t stage_id) {
  if (InferAttrs() != SUCCESS) {
    MS_LOG(ERROR) << name_ << ": Infer attrs failed";
    return FAILED;
  }

  DivisorsReplaceShapes();  // in dynamic shape, using divisors replace to shapes before CheckStrategy and so on
  std::vector<StrategyPtr> sp_vector = GenerateOpStrategies(stage_id);
  ResumeShapes();  // resume shapes

  size_t success = 0;
  for (auto &sp : sp_vector) {
    if (SetCostUnderStrategy(sp) == SUCCESS) {
      success++;
      MS_LOG(INFO) << name_ << ": Successfully generated the " << GetSerialNumberString(success)
                   << " strategy: " << sp->ToString();
    } else {
      MS_LOG(INFO) << name_ << ": SetCostUnderStrategy failed, the strategy is " << sp->ToString();
    }
  }
  return SUCCESS;
}

int64_t OperatorInfo::GetIntAttr(const std::string &attr_name) {
  auto attr_iter = attrs_.find(attr_name);
  if (attr_iter == attrs_.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Can not find the attribution of " << attr_name;
  }

  MS_EXCEPTION_IF_NULL(attr_iter->second);
  if (!attr_iter->second->isa<Int64Imm>()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": The value of " << attr_name << " is not int";
  }

  return attr_iter->second->cast<Int64ImmPtr>()->value();
}

bool OperatorInfo::GetBoolAttr(const std::string &attr_name) {
  auto attr_iter = attrs_.find(attr_name);
  if (attr_iter == attrs_.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Can not find the attribution of " << attr_name;
  }

  MS_EXCEPTION_IF_NULL(attr_iter->second);
  if (!attr_iter->second->isa<BoolImm>()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": The value of " << attr_name << " is not int";
  }

  return attr_iter->second->cast<BoolImmPtr>()->value();
}

std::string OperatorInfo::GetStringAttr(const std::string &attr_name) {
  std::string string_attr;
  auto attr_iter = attrs_.find(attr_name);
  if (attr_iter == attrs_.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Can not find the attribution of " << attr_name;
  }

  MS_EXCEPTION_IF_NULL(attr_iter->second);
  if (!attr_iter->second->isa<StringImm>()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": The value of " << attr_name << " is not string";
  }

  string_attr = attr_iter->second->cast<StringImmPtr>()->value();
  return string_attr;
}

std::vector<int64_t> OperatorInfo::GetTupleIntAttr(const std::string &attr_name) {
  std::vector<int64_t> tuple_attr;
  auto tuple_attr_iter = attrs_.find(attr_name);
  if (tuple_attr_iter == attrs_.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Can not find the attribution of " << attr_name;
  }

  MS_EXCEPTION_IF_NULL(tuple_attr_iter->second);
  tuple_attr = GetValue<std::vector<int64_t>>(tuple_attr_iter->second);

  return tuple_attr;
}

float OperatorInfo::GetFloatAttr(const std::string &attr_name) {
  auto attr_iter = attrs_.find(attr_name);
  if (attr_iter == attrs_.end()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": Can not find the attribution of " << attr_name;
  }

  MS_EXCEPTION_IF_NULL(attr_iter->second);
  if (!attr_iter->second->isa<FP32Imm>()) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_) << name_ << ": The value of " << attr_name << " is not float";
  }

  return attr_iter->second->cast<FP32ImmPtr>()->value();
}

std::vector<ValuePtr> GetValueSequence(const ValuePtr &sequence) {
  MS_EXCEPTION_IF_NULL(sequence);
  std::vector<ValuePtr> ret;
  if (!sequence->isa<ValueTuple>() && !sequence->isa<ValueList>()) {
    MS_LOG(ERROR) << "The arg is not value tuple or value list";
    return ret;
  }

  if (sequence->isa<ValueTuple>()) {
    auto val_tuple = sequence->cast<ValueTuplePtr>();
    return val_tuple->value();
  }
  auto val = sequence->cast<ValueListPtr>();
  MS_EXCEPTION_IF_NULL(val);
  return val->value();
}

ValuePtr MakeListValue(const std::vector<int64_t> &v) {
  std::vector<ValuePtr> list;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(list), [](int64_t ele) { return MakeValue(ele); });
  return std::make_shared<ValueSequence>(list);
}

ValuePtr MakeTupleListValue(const Shapes &v) {
  std::vector<ValuePtr> tuple;
  (void)std::transform(v.begin(), v.end(), std::back_inserter(tuple),
                       [](const std::vector<int64_t> &list) { return MakeListValue(list); });
  return std::make_shared<ValueTuple>(tuple);
}

AnfNodePtr CreateValueTupleAnfNodePtr(const std::vector<int64_t> &value_tuple) {
  auto value_ptr = MakeValue(value_tuple)->cast<ValueTuplePtr>();
  auto value_node = NewValueNode(value_ptr);
  return value_node->cast<AnfNodePtr>();
}

AnfNodePtr CreateTensorTupleAnfNodePtr(const tensor::TensorPtrList &tensor_tuple) {
  auto tensor_ptr = MakeValue(tensor_tuple)->cast<ValueTuplePtr>();
  auto tensor_node = NewValueNode(tensor_ptr);
  return tensor_node->cast<AnfNodePtr>();
}

Operator CreateDivOpWithType(float divisor, const TypePtr &dtype) {
  OperatorName operator1_name = REAL_DIV;
  mindspore::tensor::TensorPtr tensor_ptr = std::make_shared<mindspore::tensor::Tensor>(divisor, dtype);
  ValuePtr op1_param_value = MakeValue(tensor_ptr);
  Attr op1_param = std::make_pair("divisor", op1_param_value);
  OperatorParams operator1_params = {std::make_pair(op1_param, 2)};
  OperatorAttrs operator1_attrs;
  OperatorArgs operator1_args = std::make_pair(operator1_attrs, operator1_params);
  Operator div_op = std::make_pair(operator1_name, operator1_args);
  return div_op;
}

ForwardOp CreateReduceMeanForwardOp(const std::vector<Group> &forward_group, const TypePtr &dtype) {
  // Create AllReduceSum op
  Operator op0 = CreateAllReduceOp(REDUCE_OP_SUM, forward_group[0].name());
  std::string group_name = forward_group[0].name();
  MS_LOG(INFO) << "The group of forward all reduce is " << group_name;

  // Create RealDiv op
  std::vector<Device> device_list = forward_group[0].GetDevicesList();
  auto divisor = SizeToFloat(device_list.size());
  Operator op1 = CreateDivOpWithType(divisor, dtype);
  std::string dtype_name = dtype->ToString();
  MS_LOG(INFO) << "The divisor of Div op is " << device_list.size() << ", the dtype is " << dtype_name;

  return {op0, op1};
}

ForwardOp CreateMeanExtForwardOp(const Group &forward_group, const TypePtr &dtype) {
  // Create AllReduceSum op
  Operator op0 = CreateAllReduceOp(REDUCE_OP_SUM, forward_group.name());
  std::string group_name = forward_group.name();
  MS_LOG(INFO) << "The group of forward all reduce is " << group_name;

  // Create RealDiv op
  std::vector<Device> device_list = forward_group.GetDevicesList();
  auto divisor = SizeToFloat(device_list.size());
  Operator op1 = CreateDivOpWithType(divisor, dtype);
  std::string dtype_name = dtype->ToString();
  MS_LOG(INFO) << "The divisor of Div op is " << device_list.size() << ", the dtype is " << dtype_name;

  return {op0, op1};
}

std::vector<int64_t> GetTensorValue(const ValuePtr &ori_value) {
  MS_EXCEPTION_IF_NULL(ori_value);
  if (!ori_value->isa<tensor::Tensor>()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Value is not tensor";
  }
  auto tensor_ptr = ori_value->cast<tensor::TensorPtr>();
  std::vector<int64_t> value;
  auto element_size = tensor_ptr->data().size();
  auto *data = static_cast<int64_t *>(tensor_ptr->data_c());
  for (auto i = 0; i < element_size; i++) {
    value.push_back(data[i]);
  }
  return value;
}

Dimensions ConvertLayoutToDemensions(const Shape &dev_matrix, const std::vector<Shape> &tensor_map) {
  Dimensions dimens;
  for (size_t i = 0; i < tensor_map.size(); ++i) {
    int64_t accu_shp = 1;
    for (size_t j = 0; j < tensor_map[i].size(); ++j) {
      if (tensor_map[i][j] == -1) {
        continue;
      }
      size_t tensor_index = dev_matrix.size() - 1 - static_cast<size_t>(tensor_map[i][j]);
      auto shard_size = dev_matrix[tensor_index];
      accu_shp *= shard_size;
    }
    dimens.push_back(accu_shp);
  }
  return dimens;
}

Status OperatorInfo::AddSwcUnderPrevOpDevMatrixSingle(const Shape &prev_op_dev_matrix,
                                                      const std::vector<Shape> &prev_op_tensor_map,
                                                      size_t layout_index) {
  if (prev_op_dev_matrix.empty()) {
    MS_LOG(INFO) << "Layout propagation prev_op_dev_matrix is empty";
    return FAILED;
  }
  if (inputs_shape_.size() <= layout_index) {
    MS_LOG(INFO) << "layout_index is illegal:" << layout_index << " while inputs_shape_ size:" << inputs_shape_.size();
    return FAILED;
  }
  if (inputs_shape_[layout_index].size() != prev_op_tensor_map.size()) {
    MS_LOG(INFO) << "prev_op_tensor_map is illegal";
    return FAILED;
  }
  if (dev_matrix_shape_ == prev_op_dev_matrix) {
    return SUCCESS;
  }
  if (strategy_cost_.empty()) {
    return SUCCESS;
  }

  dev_matrix_shape_ = prev_op_dev_matrix;
  std::vector<StrategyPtr> strategy_ptrs;
  (void)std::transform(strategy_cost_.begin(), strategy_cost_.end(), std::back_inserter(strategy_ptrs),
                       [](const auto &swc) { return swc->strategy_ptr; });
  size_t add_cnt = 0;
  for (const auto &strategy_ptr : strategy_ptrs) {
    if (strategy_ptr->GetInputDim()[layout_index] !=
        ConvertLayoutToDemensions(prev_op_dev_matrix, prev_op_tensor_map)) {
      continue;
    }
    std::vector<std::shared_ptr<TensorLayout>> in_tensor_layouts =
      InferLayoutsByStrategy(strategy_ptr, prev_op_tensor_map, layout_index);
    if (in_tensor_layouts.empty()) {
      continue;
    }
    auto out_tensor_layouts = std::vector<std::shared_ptr<TensorLayout>>();
    if (SetCostUnderLayout(strategy_ptr, nullptr, in_tensor_layouts, out_tensor_layouts) != SUCCESS) {
      MS_LOG(WARNING) << "Failure: operator " << name_ << " SetCostUnderLayout failed";
      return FAILED;
    }
    add_cnt++;
  }
  MS_LOG(INFO) << name_ << " add " << add_cnt << " swcs.";
  return SUCCESS;
}

Status OperatorInfo::AddSwcUnderNextOpDevMatrixSingle(const std::shared_ptr<OperatorInfo> &next_op,
                                                      const std::shared_ptr<Edge> &edge) {
  const Shape &next_op_dev_matrix = next_op->dev_matrix_shape();

  if (next_op_dev_matrix.empty()) {
    MS_LOG(WARNING) << "Layout propagation next_op_dev_matrix is empty";
    return FAILED;
  }

  const TensorMapBefores &next_op_inputs_tensor_map_before = next_op->inputs_tensor_map_before();
  if (next_op_inputs_tensor_map_before.empty()) {
    MS_LOG(WARNING) << "Layout propagation next_op_inputs_tensor_map_before is empty";
    return FAILED;
  }

  const std::vector<Shape> &next_op_tensor_map = next_op_inputs_tensor_map_before[edge->next_op_input_index()];

  size_t layout_index = kIndex0;

  if (inputs_shape_.size() <= layout_index) {
    MS_LOG(WARNING) << "layout_index is illegal:" << layout_index
                    << " while inputs_shape_ size:" << inputs_shape_.size();
    return FAILED;
  }

  if (inputs_shape_[layout_index].size() != next_op_tensor_map.size()) {
    MS_LOG(INFO) << "next_op_tensor_map is consistent";
    return FAILED;
  }

  if (strategy_cost_.empty()) {
    return SUCCESS;
  }

  dev_matrix_shape_ = next_op_dev_matrix;
  std::vector<StrategyPtr> strategy_ptrs;
  (void)std::transform(strategy_cost_.begin(), strategy_cost_.end(), std::back_inserter(strategy_ptrs),
                       [](const auto &swc) { return swc->strategy_ptr; });
  size_t add_cnt = 0;
  for (const auto &strategy_ptr : strategy_ptrs) {
    if (strategy_ptr->GetInputDim()[layout_index] !=
        ConvertLayoutToDemensions(next_op_dev_matrix, next_op_tensor_map)) {
      continue;
    }
    std::vector<std::shared_ptr<TensorLayout>> in_tensor_layouts =
      InferLayoutsByStrategy(strategy_ptr, next_op_tensor_map, layout_index);
    if (in_tensor_layouts.empty()) {
      continue;
    }
    auto out_tensor_layouts = std::vector<std::shared_ptr<TensorLayout>>();
    if (SetCostUnderLayout(strategy_ptr, nullptr, in_tensor_layouts, out_tensor_layouts) != SUCCESS) {
      MS_LOG(WARNING) << "Failure: operator " << name_ << " SetCostUnderLayout failed";
      return FAILED;
    }
    MS_LOG(INFO) << "op: " << name_ << " add swc, in tensor layout: " << in_tensor_layouts[kIndex0]->ToString();
    add_cnt++;
  }
  MS_LOG(INFO) << name_ << " add " << add_cnt << " swcs.";
  return SUCCESS;
}

std::vector<std::shared_ptr<TensorLayout>> OperatorInfo::InferLayoutsByStrategy(
  const StrategyPtr &strategy_ptr, const std::vector<Shape> &prev_op_tensor_map, size_t layout_index) {
  std::vector<std::shared_ptr<TensorLayout>> in_layouts;
  Strategies inputs = strategy_ptr->GetInputDim();
  for (size_t i = 0; i < inputs.size(); i++) {
    auto in_layout = std::make_shared<TensorLayout>();
    if (i == layout_index) {
      if (in_layout->InitFromExtendVector(dev_matrix_shape_, prev_op_tensor_map, inputs_shape_[i]) != SUCCESS) {
        MS_LOG(WARNING) << "InferLayoutsByStrategy failed.";
        return std::vector<std::shared_ptr<TensorLayout>>();
      }
      in_layouts.push_back(in_layout);
      continue;
    }
    std::vector<Shape> empty_tensor_map;
    if (!inputs[i].empty()) {
      MS_LOG(WARNING) << "Not Support Multiple inputs operator";
      return std::vector<std::shared_ptr<TensorLayout>>();
    }
    if (in_layout->InitFromExtendVector(dev_matrix_shape_, empty_tensor_map, inputs_shape_[i]) != SUCCESS) {
      MS_LOG(WARNING) << "InferLayoutsByStrategy failed.";
      return std::vector<std::shared_ptr<TensorLayout>>();
    }
    in_layouts.push_back(in_layout);
  }
  return in_layouts;
}

bool OperatorInfo::CheckPrevOpStatus(const Shape &prev_op_dev_matrix, const std::vector<Shape> &prev_op_tensor_map,
                                     size_t layout_index) {
  if (prev_op_dev_matrix.empty()) {
    MS_LOG(INFO) << "Layout propagation prev_op_dev_matrix is empty";
    return false;
  }
  if (inputs_shape_.size() <= layout_index) {
    MS_LOG(INFO) << "layout_index is illegal:" << layout_index << " while inputs_shape_ size:" << inputs_shape_.size();
    return false;
  }
  if (inputs_shape_[layout_index].size() != prev_op_tensor_map.size()) {
    MS_LOG(INFO) << "prev_op_tensor_map is illegal";
    return false;
  }

  return true;
}

bool OperatorInfo::StrategyMatchTensorMap(const StrategyPtr &strategy_ptr,
                                          const std::vector<std::vector<Shape>> &prev_op_tensor_maps) {
  for (size_t i = 0; i < prev_op_tensor_maps.size(); i++) {
    if (strategy_ptr->GetInputDim()[i] != ConvertLayoutToDemensions(dev_matrix_shape_, prev_op_tensor_maps[i])) {
      return false;
    }
  }
  return true;
}

Status OperatorInfo::AddSwcUnderPrevOpDevMatrixMulti() {
  std::vector<Shape> prev_op_dev_matrixs;
  std::vector<std::vector<Shape>> prev_op_tensor_maps(inputs_shape_.size());

  auto it = visited_edges_.begin();
  const auto &prev_operator_first = (*it)->prev_operator();
  if (prev_operator_first == nullptr) {
    MS_LOG(WARNING) << "prev operator is null";
    return FAILED;
  }
  const Shape &dev_matrix_first = prev_operator_first->out_dev_matrix_shape();
  size_t layout_index_first = (*it)->next_op_input_index();

  if (prev_operator_first->outputs_tensor_map_before().empty()) {
    MS_LOG(WARNING) << "Add swcs under device matrix failed, prev_op_tensor_map empty, op: "
                    << prev_operator_first->name();
    return FAILED;
  }
  const std::vector<Shape> prev_op_tensor_map_first =
    prev_operator_first->outputs_tensor_map_before()[(*it)->prev_op_output_index()];
  if (!CheckPrevOpStatus(dev_matrix_first, prev_op_tensor_map_first, layout_index_first)) {
    return FAILED;
  }
  prev_op_tensor_maps[layout_index_first] = prev_op_tensor_map_first;
  it++;

  while (it != visited_edges_.end()) {
    const auto &prev_operator = (*it)->prev_operator();
    if (prev_operator == nullptr) {
      MS_LOG(WARNING) << "prev operator is null";
      return FAILED;
    }
    const Shape &dev_matrix = prev_operator->out_dev_matrix_shape();
    size_t layout_index = (*it)->next_op_input_index();

    if (dev_matrix != dev_matrix_first) {
      MS_LOG(WARNING) << "Add swcs under device matrix failed, prev ops dev_matrix inconsistent, op: "
                      << prev_operator->name();
      return FAILED;
    }

    if (prev_operator->outputs_tensor_map_before().empty()) {
      MS_LOG(WARNING) << "Add swcs under device matrix failed, prev_op_tensor_map empty, op: " << prev_operator->name();
      return FAILED;
    }

    const std::vector<Shape> prev_op_tensor_map =
      prev_operator->outputs_tensor_map_before()[(*it)->prev_op_output_index()];
    if (!CheckPrevOpStatus(dev_matrix, prev_op_tensor_map, layout_index)) {
      return FAILED;
    }
    prev_op_tensor_maps[layout_index] = prev_op_tensor_map;
    it++;
  }

  if (strategy_cost_.empty()) {
    return SUCCESS;
  }

  dev_matrix_shape_ = dev_matrix_first;
  std::vector<StrategyPtr> strategy_ptrs;
  (void)std::transform(strategy_cost_.begin(), strategy_cost_.end(), std::back_inserter(strategy_ptrs),
                       [](const auto &swc) { return swc->strategy_ptr; });
  size_t add_cnt = 0;
  for (const auto &strategy_ptr : strategy_ptrs) {
    if (!StrategyMatchTensorMap(strategy_ptr, prev_op_tensor_maps)) {
      continue;
    }

    MS_LOG(INFO) << "Add swcs under device matrix find valid strategy: " << strategy_ptr->ToString();

    std::vector<std::shared_ptr<TensorLayout>> in_tensor_layouts =
      InferLayoutsByStrategy(strategy_ptr, prev_op_tensor_maps);
    if (in_tensor_layouts.empty()) {
      continue;
    }
    auto out_tensor_layouts = std::vector<std::shared_ptr<TensorLayout>>();
    if (SetCostUnderLayout(strategy_ptr, nullptr, in_tensor_layouts, out_tensor_layouts) != SUCCESS) {
      MS_LOG(WARNING) << "Failure: operator " << name_ << " SetCostUnderLayout failed";
      return FAILED;
    }
    add_cnt++;
  }
  MS_LOG(INFO) << name_ << " add " << add_cnt << " swcs.";
  return SUCCESS;
}

std::vector<std::shared_ptr<TensorLayout>> OperatorInfo::InferLayoutsByStrategy(
  const StrategyPtr &strategy_ptr, const std::vector<std::vector<Shape>> &prev_op_tensor_maps) {
  std::vector<std::shared_ptr<TensorLayout>> in_layouts;
  Strategies inputs = strategy_ptr->GetInputDim();

  for (size_t i = 0; i < inputs.size(); i++) {
    auto in_layout = std::make_shared<TensorLayout>();
    std::vector<Shape> tensor_map;
    if (!inputs[i].empty()) {
      tensor_map = prev_op_tensor_maps[i];
    }

    if (in_layout->InitFromExtendVector(dev_matrix_shape_, tensor_map, inputs_shape_[i]) != SUCCESS) {
      MS_LOG(WARNING) << "InferLayoutsByStrategy failed.";
      return std::vector<std::shared_ptr<TensorLayout>>();
    }
    in_layouts.push_back(in_layout);
    MS_LOG(INFO) << "InferLayoutsByStrategy in_layout: " << in_layout->ToString();
  }
  return in_layouts;
}

void OperatorInfo::InitVisitedEdges() {
  for (auto &edge : visited_edges_) {
    if (edge->InitEdgeCost() != SUCCESS) {
      MS_LOG(EXCEPTION) << "Edge cost initialization failed.";
    }
  }
  return;
}

}  // namespace parallel
}  // namespace mindspore
