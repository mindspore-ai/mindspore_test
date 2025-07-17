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

#include "frontend/parallel/ops_info/apply_rotary_pos_emb_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
// case1:
// ApplyRotaryPosEmb has 5 inputs and 2 outputs
// query:         (batch , seq_len (inc is 1), query_hidden_size)
// key:           (batch,  seq_len (inc is 1), key_hidden_size)
// cos:           (max_seq_len, head_dim)
// sin:           (max_seq_len, head_dim)
// position_ids:  (max_seq_len or batch)
// ------------------------------
// output_query:  (batch, seq_len (inc is 1), query_hidden_size)
// output_key:    (batch, seq_len (inc is 1), key_hidden_size)

// split strategy
// batch is not able to split
// seq_len is not able to split
// query_hidden_size is able to split
// key_hidden_size is able to split
// if inc, position_ids is able to split

// case2:
// ApplyRotaryPosEmb has 5 inputs and 2 outputs
// query:         (dp, 1, mp, 1)
// key:           (dp, 1, mp, 1)
// cos:           (dp, 1, 1, 1)
// sin:           (dp, 1, 1, 1)
// position_ids:  (1)
// ------------------------------
// output_query:  (dp, 1, mp, 1)
// output_key:    (dp, 1, mp, 1)

constexpr size_t kApplyRotaryOutputSize = 2;
constexpr size_t kApplyRotaryPosEmbQueryIndex = 0;
constexpr size_t kApplyRotaryPosEmbKeyIndex = 1;
constexpr size_t kApplyRotaryPosEmbCosIndex = 2;
constexpr size_t kApplyRotaryPosEmbSinIndex = 3;
constexpr size_t kApplyRotaryPosEmbPositionIdsIndex = 4;
constexpr size_t kInputQueryBatchIndex = 0;
constexpr size_t kInputQuerySeqLenIndex = 1;
constexpr size_t kInputQueryHiddenSizeIndex = 2;
constexpr size_t kInputQueryNumIndex = 2;
constexpr size_t kInputQueryDimIndex = 3;
constexpr size_t kInputKeyBatchIndex = 0;
constexpr size_t kInputKeySeqLenIndex = 1;
constexpr size_t kInputKeyHiddenSizeIndex = 2;
constexpr size_t kInputKeyNumIndex = 2;
constexpr size_t kInputKeyDimIndex = 3;
constexpr size_t kInputCosSeqLenIndex = 0;
constexpr size_t kInputCosHeaDimIndex = 1;
constexpr size_t kInputSinSeqLenIndex = 0;
constexpr size_t kInputSinHeaDimIndex = 1;
constexpr size_t kInputPositionIdsBatchIndex = 0;
constexpr size_t kIncInferSeqLen = 1;

Status ApplyRotaryPosEmbInfo::SetDims(const StrategyPtr &strategy) {
  auto input_strategys = strategy->GetInputDim();
  auto strategy_cache = input_strategys.at(0);

  const size_t input_dims4 = 4;
  const size_t input_dims3 = 3;
  if (strategy_cache.size() == input_dims4) {
    is_input_dims_4_ = true;
  } else if (strategy_cache.size() == input_dims3) {
    is_input_dims_4_ = false;
  } else {
    return FAILED;
  }

  return SUCCESS;
}
Status ApplyRotaryPosEmbInfo::CheckStrategy3Dims(const Dimensions &strategy_query, const Dimensions &strategy_key,
                                                 const Dimensions &strategy_position_ids) {
  if (strategy_query.at(kInputQuerySeqLenIndex) != 1 || strategy_key.at(kInputKeySeqLenIndex) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The seq_len can't be shard, but got"
                  << " query's seq_len strategy: " << strategy_query.at(kInputQuerySeqLenIndex)
                  << ", key's seq_len strategy: " << strategy_key.at(kInputKeySeqLenIndex);
    return FAILED;
  }

  if ((strategy_query.at(kInputQueryHiddenSizeIndex) != strategy_key.at(kInputKeyHiddenSizeIndex))) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The hidden_size must be shard at the same time, but got"
                  << " query's strategy: " << strategy_query << ", key's strategy: " << strategy_key;
    return FAILED;
  }

  if ((strategy_query.at(kInputQueryBatchIndex) != strategy_key.at(kInputKeyBatchIndex)) ||
      (strategy_query.at(kInputQueryBatchIndex) != strategy_position_ids.at(kInputPositionIdsBatchIndex))) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The batch must be shard at the same time, but got"
                  << " query's strategy: " << strategy_query << ", key's strategy: " << strategy_key
                  << ", position_ids's strategy: " << strategy_position_ids;
    return FAILED;
  }
  return SUCCESS;
}

Status ApplyRotaryPosEmbInfo::CheckStrategy4Dims(const Dimensions &strategy_query, const Dimensions &strategy_key) {
  if (strategy_query.at(kInputQuerySeqLenIndex) != 1 || strategy_key.at(kInputKeySeqLenIndex) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The seq_len can't be shard, but got"
                  << " query's seq_len strategy: " << strategy_query.at(kInputQuerySeqLenIndex)
                  << ", key's seq_len strategy: " << strategy_key.at(kInputKeySeqLenIndex);
    return FAILED;
  }

  if (strategy_query.at(kInputQueryDimIndex) != 1 || strategy_key.at(kInputKeyDimIndex) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The dim can't be shard, but got"
                  << " query's dim strategy: " << strategy_query.at(kInputQueryDimIndex)
                  << ", key's dim strategy: " << strategy_key.at(kInputKeyDimIndex);
    return FAILED;
  }

  if ((strategy_query.at(kInputQueryNumIndex) != strategy_key.at(kInputKeyNumIndex))) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The num must be shard at the same time, but got"
                  << " query's strategy: " << strategy_query << ", key's strategy: " << strategy_key;
    return FAILED;
  }

  if ((strategy_query.at(kInputQueryBatchIndex) != strategy_key.at(kInputKeyBatchIndex))) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The batch must be shard at the same time, but got"
                  << " query's strategy: " << strategy_query << ", key's strategy: " << strategy_key;
    return FAILED;
  }
  return SUCCESS;
}

Status ApplyRotaryPosEmbInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  if (SetDims(strategy) != SUCCESS) {
    return FAILED;
  }
  auto input_strategies = strategy->GetInputDim();
  auto strategy_query = input_strategies.at(kApplyRotaryPosEmbQueryIndex);
  auto strategy_key = input_strategies.at(kApplyRotaryPosEmbKeyIndex);
  auto strategy_cos = input_strategies.at(kApplyRotaryPosEmbCosIndex);
  auto strategy_sin = input_strategies.at(kApplyRotaryPosEmbSinIndex);
  auto strategy_position_ids = input_strategies.at(kApplyRotaryPosEmbPositionIdsIndex);  // (dp)

  if (strategy_cos.at(kInputCosSeqLenIndex) != 1 || strategy_cos.at(kInputCosHeaDimIndex) != 1 ||
      strategy_sin.at(kInputSinSeqLenIndex) != 1 || strategy_cos.at(kInputSinHeaDimIndex) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The cos and sin can't be shard, but got"
                  << " cos's strategy: " << strategy_cos << ", sin's strategy: " << strategy_sin;
    return FAILED;
  }
  if (is_input_dims_4_) {
    return CheckStrategy4Dims(strategy_query, strategy_key);
  }
  return CheckStrategy3Dims(strategy_query, strategy_key, strategy_position_ids);
}

Status ApplyRotaryPosEmbInfo::InferDevMatrixShape() {
  auto input_strategies = strategy()->GetInputDim();
  auto query = input_strategies.at(0);
  // mp   dp
  // 1    0
  if (is_input_dims_4_) {
    dev_matrix_shape_ = {query.at(kInputQueryNumIndex), query.at(kInputQueryBatchIndex)};
  } else {
    dev_matrix_shape_ = {query.at(kInputQueryHiddenSizeIndex), query.at(kInputQueryBatchIndex)};
  }
  return SUCCESS;
}

Status ApplyRotaryPosEmbInfo::InferTensorMap() {
  if (is_input_dims_4_) {
    Shape query_tensor_map{0, -1, 1, -1};
    Shape key_tensor_map{0, -1, 1, -1};
    Shape cos_tensor_map{0, -1, -1, -1};
    Shape sin_tensor_map{0, -1, -1, -1};
    Shape position_ids_tensor_map;
    position_ids_tensor_map.push_back(-1);

    (void)inputs_tensor_map_.emplace_back(query_tensor_map);
    (void)inputs_tensor_map_.emplace_back(key_tensor_map);
    (void)inputs_tensor_map_.emplace_back(cos_tensor_map);
    (void)inputs_tensor_map_.emplace_back(sin_tensor_map);
    (void)inputs_tensor_map_.emplace_back(position_ids_tensor_map);

    Shape output_query_tensor_map{0, -1, 1, -1};
    Shape output_key_tensor_map{0, -1, 1, -1};
    outputs_tensor_map_.emplace_back(output_query_tensor_map);
    outputs_tensor_map_.emplace_back(output_key_tensor_map);
  } else {
    Shape query_tensor_map{0, -1, 1};
    Shape key_tensor_map{0, -1, 1};
    Shape cos_tensor_map{-1, -1};
    Shape sin_tensor_map{-1, -1};
    auto input_position_ids_shape = inputs_shape_.at(kApplyRotaryPosEmbPositionIdsIndex);
    auto input_position_ids_shape_value = input_position_ids_shape.at(0);
    Shape position_ids_tensor_map;
    if (input_position_ids_shape_value == kIncInferSeqLen) {
      // INC infer
      position_ids_tensor_map.push_back(0);
    } else {
      position_ids_tensor_map.push_back(-1);
    }

    (void)inputs_tensor_map_.emplace_back(query_tensor_map);
    (void)inputs_tensor_map_.emplace_back(key_tensor_map);
    (void)inputs_tensor_map_.emplace_back(cos_tensor_map);
    (void)inputs_tensor_map_.emplace_back(sin_tensor_map);
    (void)inputs_tensor_map_.emplace_back(position_ids_tensor_map);

    Shape output_query_tensor_map{0, -1, 1};
    Shape output_key_tensor_map{0, -1, 1};
    outputs_tensor_map_.emplace_back(output_query_tensor_map);
    outputs_tensor_map_.emplace_back(output_key_tensor_map);
  }
  return SUCCESS;
}

Status ApplyRotaryPosEmbInfo::InferAsLossDivisor() {
  if (outputs_tensor_map_.size() != kApplyRotaryOutputSize) {
    MS_LOG(ERROR) << name_ << ": The size of outputs tensor map must be 2, but got " << outputs_tensor_map_.size();
    return FAILED;
  }
  as_loss_divisor_ = ComputeRepeatDeviceNumByTensorMap(dev_matrix_shape_, outputs_tensor_map_[0]);
  MS_LOG(INFO) << name_ << " : The dev matrix shape is " << ShapeToString(dev_matrix_shape_)
               << ", the output[0]'s tensor map is " << ShapeToString(outputs_tensor_map_[0])
               << ", as_loss_divisor_ is " << as_loss_divisor_;
  return SUCCESS;
}
REGISTER(ApplyRotaryPosEmbInfo);
}  // namespace parallel
}  // namespace mindspore
