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

#include "frontend/parallel/ops_info/paged_attention_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
// PagedAttention has 9 inputs
// query:           (batch, seq_len, hidden_size)
// key_cache:       (block_size, num_blocks, num_head, head_dim)
// value_cache:     (block_size, num_blocks, num_head, head_dim)
// block_tables:    (batch, max_num_block_per_batch)
// context_lens:    (batch * seq_len)
// antiquant_scale: (2, num_head * head_dim)
// antiquant_offset:(2, num_head * head_dim)
// attn_mask:       (num_tokens, max_kv_seq_len)
// q_seq_lens:      (batch, )
// ------------------------------
// output:          (batch, seq_len, hidden_size)

// split strategy
// num_blocks is not able to split
// block_size is not able to split
// batch is able to split
// hidden_size is able to split
// num_head is able to split
// head_dim is able to split
// max_num_block_per_batch is not able to split

Status PagedAttentionInfo::CheckStrategy(const StrategyPtr &strategy) {
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }

  auto input_strategies = strategy->GetInputDim();
  auto strategy_query = input_strategies.at(0);         // (dp, 1, mp)
  auto strategy_key_cache = input_strategies.at(1);     // (1, 1, mp, 1)
  auto strategy_value_cache = input_strategies.at(2);   // (1, 1, mp, 1)
  auto strategy_block_tables = input_strategies.at(3);  // (dp, 1)

  if (strategy_block_tables.at(1) != 1) {
    MS_LOG(ERROR)
      << name_
      << ": Invalid strategy: The second dim of block_tables \"max_num_block_per_batch\" can't be shard, but got"
      << " block_tables's strategy: " << strategy_block_tables;
    return FAILED;
  }

  if (strategy_key_cache.at(0) != 1 || strategy_value_cache.at(0) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The block_size can't be shard, but got"
                  << " key_cache's block_size strategy: " << strategy_key_cache.at(0)
                  << ", value_cache's block_size strategy: " << strategy_value_cache.at(0);
    return FAILED;
  }

  if (strategy_key_cache.at(1) != 1 || strategy_value_cache.at(1) != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The num_blocks can't be shard, but got"
                  << " key_cache's num_blocks strategy: " << strategy_key_cache.at(1)
                  << ", value_cache's num_blocks strategy: " << strategy_value_cache.at(1);
    return FAILED;
  }

  if ((strategy_key_cache.at(2) != strategy_value_cache.at(2)) || (strategy_query.at(2) != strategy_key_cache.at(2))) {
    MS_LOG(ERROR)
      << name_
      << ": Invalid strategy: The cache num_head and update hidden_size must be shard at the same time, but got"
      << " query's strategy: " << strategy_query << ", key_cache's strategy: " << strategy_key_cache
      << ", value_cache's strategy: " << strategy_value_cache;
    return FAILED;
  }

  if (strategy_key_cache.size() == 4 &&
      ((strategy_key_cache.at(3) != strategy_value_cache.at(3)) || (strategy_key_cache.at(3) != 1))) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The head_dim can't be shard, but got"
                  << " query's strategy: " << strategy_query << ", key_cache's strategy: " << strategy_key_cache
                  << ", value_cache's strategy: " << strategy_value_cache;
    return FAILED;
  }

  return SUCCESS;
}

Status PagedAttentionInfo::InferDevMatrixShape() {
  auto input_strategies = strategy()->GetInputDim();
  auto query = input_strategies.at(0);  // (batch, seq_len, hidden_size)
  auto cache = input_strategies.at(1);  // (block_size, num_blocks, hidden_size)

  // batch   block_size   num_blocks   seq_len   hidden_size
  //  4          3             2          1          0
  dev_matrix_shape_ = {query.at(0), cache.at(0), cache.at(1), query.at(1), cache.at(2)};

  return SUCCESS;
}

Status PagedAttentionInfo::InferTensorMap() {
  auto input_strategies = strategy()->GetInputDim();

  optional_inputs_.resize(input_value_.size(), true);
  for (size_t index = 0; index < input_value_.size(); index++) {
    auto optional_input_ptr = input_value_[index];
    if (optional_input_ptr != nullptr && optional_input_ptr->isa<None>()) {
      optional_inputs_[index] = false;
    }
  }
  const int kQuantScaleIndex = 5;
  const int kQuantOffsetIndex = 6;
  const int kAttenMaskIndex = 7;
  const int kQLensIndex = 8;

  Shape query_tensor_map{4, 1, 0};
  Shape cache_tensor_map{-1, -1, 0};
  auto cache = input_strategies.at(2);
  if (cache.size() == 4) {
    cache_tensor_map.push_back(-1);
  }
  Shape block_tensor_map{4, -1};
  Shape context_tensor_map{-1};

  inputs_tensor_map_.emplace_back(query_tensor_map);
  inputs_tensor_map_.emplace_back(cache_tensor_map);
  inputs_tensor_map_.emplace_back(cache_tensor_map);
  inputs_tensor_map_.emplace_back(block_tensor_map);
  inputs_tensor_map_.emplace_back(context_tensor_map);

  const int kQuantModeIndex = 12;
  const int64_t per_token_quant_mode = 1;
  auto quant_mode = GetValue<int64_t>(input_value_[kQuantModeIndex]);
  Shape antiquant_tensor_map;
  if (quant_mode == per_token_quant_mode) {
    antiquant_tensor_map = {-1, -1, -1};
  } else {
    antiquant_tensor_map = {-1, 0};
  }

  if (optional_inputs_[kQuantScaleIndex]) {
    inputs_tensor_map_.emplace_back(antiquant_tensor_map);
  }
  if (optional_inputs_[kQuantOffsetIndex]) {
    inputs_tensor_map_.emplace_back(antiquant_tensor_map);
  }
  if (optional_inputs_[kAttenMaskIndex]) {
    auto atten_mask_shape_size = input_strategies[inputs_tensor_map_.size()].size();
    Shape atten_mask_map{-1, -1};
    const size_t kAttenMaskSize = 3;
    if (atten_mask_shape_size == kAttenMaskSize) {
      const size_t kAttenMaskDim4 = 4;
      atten_mask_map.insert(atten_mask_map.begin(), kAttenMaskDim4);
    }
    inputs_tensor_map_.emplace_back(atten_mask_map);
  }
  if (optional_inputs_[kQLensIndex]) {
    Shape qlen_tensor_map{-1};
    inputs_tensor_map_.emplace_back(qlen_tensor_map);
  }
  Shape out_tensor_map{4, 1, 0};
  outputs_tensor_map_.emplace_back(out_tensor_map);

  return SUCCESS;
}
REGISTER(PagedAttentionInfo);
}  // namespace parallel
}  // namespace mindspore
