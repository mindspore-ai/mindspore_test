/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_INDEX_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_INDEX_INFO_H_

#include <string>
#include <memory>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class IndexInfo : public OperatorInfo {
 public:
  IndexInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<IndexCost>()) {}
  ~IndexInfo() override = default;

  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;
  NewStrategies param_strategy_;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override { return {}; }

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status CheckIndex();
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferBias();
  RankList GetAllReduceRankList();
  std::string InferGroup();
  Status IndexShapeCheck();
  Status InferAsLossDivisor();
  Group group_;
  Status InferForwardCommunication() override { return SUCCESS; }
  std::vector<int64_t> slice_size_;
  void SetOptionalInputTensorMap(const size_t &index, size_t *valid_input_index);
  ReplaceGraphPtr ReplaceGraphDynamicShape(const CNodePtr &cnode);

 private:
  std::vector<int64_t> kernel_size_;
  std::vector<int64_t> bias_;
  std::vector<int64_t> rank_detail_;
  int64_t shard_input_data_height_;
  int64_t shard_input_data_weight_;
  Shape origin_dev_matrix_shape_;
};
}  // namespace parallel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_INDEX_INFO_H_
