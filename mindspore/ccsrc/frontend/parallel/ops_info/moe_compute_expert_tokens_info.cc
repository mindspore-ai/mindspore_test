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

#include "frontend/parallel/ops_info/moe_compute_expert_tokens_info.h"
#include "frontend/parallel/dynamic_creator.h"

namespace mindspore {
namespace parallel {
namespace {
constexpr size_t kSortedExpertIdx = 0;
constexpr size_t kInputNums = 1;
}  // namespace
Status MoeComputeExpertTokensInfo::GetAttrs() {
  // sorted_expert: tensor, num_experts: int
  std::string op_name = GetPrimNameFromInfoName(this->name_);
  std::optional<int64_t> num_expert_opt_v = GetScalarValueFromInputs<int64_t>(input_value_, op_name, "num_expert");
  if (!num_expert_opt_v.has_value()) {
    MS_LOG(ERROR) << name_ << ": Can not find the attr of num_expert.";
    return FAILED;
  }
  this->num_expert_ = num_expert_opt_v.value();
  return SUCCESS;
}

Status MoeComputeExpertTokensInfo::CheckStrategy(const StrategyPtr &strategy) {
  MS_EXCEPTION_IF_NULL(strategy);
  const auto &strategies = strategy->GetInputDim();
  if (strategies.size() != kInputNums) {
    MS_LOG(ERROR) << name_ << ": strategies count must be 1.";
    return FAILED;
  }
  if (CheckStrategyValue(strategy, inputs_shape_) != SUCCESS) {
    return FAILED;
  }
  auto strategy_sorted_experts = strategies[kSortedExpertIdx];

  if (strategy_sorted_experts[0] != 1) {
    MS_LOG(ERROR) << name_ << ": Invalid strategy: The token num can't be shard, but got"
                  << " shard num : " << strategy_sorted_experts[0];
    return FAILED;
  }

  return SUCCESS;
}

Status MoeComputeExpertTokensInfo::InferDevMatrixShape() {
  MS_EXCEPTION_IF_NULL(strategy_);
  const auto &strategies = strategy_->GetInputDim();
  if (strategies.empty()) {
    MS_LOG(ERROR) << name_ << ": The strategy can not be empty";
    return FAILED;
  }
  dev_matrix_shape_ = strategies[kSortedExpertIdx];
  return SUCCESS;
}

Status MoeComputeExpertTokensInfo::InferTensorMap() {
  // sorted_expert_strategy: (n)
  // output_tokens_strategy: (n)
  TensorMap sorted_expert_tensor_map{0};
  TensorMap expert_tokens_tensor_map{0};
  inputs_tensor_map_.emplace_back(sorted_expert_tensor_map);
  outputs_tensor_map_.emplace_back(expert_tokens_tensor_map);

  return SUCCESS;
}

REGISTER(MoeComputeExpertTokensInfo);
}  // namespace parallel
}  // namespace mindspore
