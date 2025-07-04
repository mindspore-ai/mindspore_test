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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_KLDIV_LOSS_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_KLDIV_LOSS_INFO_H_

#include <memory>
#include <vector>
#include <string>

#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"
#include "frontend/parallel/tensor_layout/tensor_redistribution.h"

namespace mindspore {
namespace parallel {
class KLDivLossInfo : public OperatorInfo {
 public:
  KLDivLossInfo(const std::string &name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                const PrimitiveAttrs &attrs)
      : OperatorInfo(name, inputs_shape, outputs_shape, attrs, std::make_shared<KLDivLossCost>()) {}
  ~KLDivLossInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override { return SetCostUnderStrategyBase(strategy); }
  void ReComputeBatchSplitFlagList() override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferForwardCommunication() override;

 private:
  Status InferGroup();

  std::size_t batch_split_num_;
  std::string reduction_;
  std::vector<Group> group_list_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_KLDIV_LOSS_INFO_H_
