/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_TILE_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_TILE_INFO_H_

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
class TileInfo : public OperatorInfo {
 public:
  TileInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
           const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<TileCost>()) {}
  ~TileInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  std::shared_ptr<Strategies> GenerateBatchStrategies() override;
  void UpdateMultiples();
  void ReplaceNodeInputOrAttrs() override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status CheckInputLayout() override;
  Status CheckOutputLayout() override;
  Status InferOutputTensorInfo() override;
  Status InferForwardCommunicationByLayout() override { return SUCCESS; }

 private:
  std::vector<int64_t> full_multiples_;
  std::vector<int64_t> slice_multiples_;
  void UpdateDynamicMultiples(const AnfNodePtr &multiples_input_node);
  TensorLayout InferOutputLayout();
};

using TileInfoPtr = std::shared_ptr<TileInfo>;
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_TILE_INFO_H_
