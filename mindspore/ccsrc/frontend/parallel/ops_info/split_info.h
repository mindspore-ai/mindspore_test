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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SPLIT_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SPLIT_INFO_H_

#include <string>
#include <memory>
#include <vector>

#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class SplitInfo : public OperatorInfo {
 public:
  SplitInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
            const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<SplitCost>()) {}
  ~SplitInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  std::shared_ptr<Strategies> GenerateBatchStrategies() override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferAsLossDivisor() override;
  Status InferMirrorOps() override;
  Status CheckInputLayout() override;
  Status CheckOutputLayout() override;
  Status InferOutputTensorInfo() override;
  void UpdateOutputTensorInfoForInterleaved() override;
  virtual Status ComputeReplaceGraphForInterleaved(const CNodePtr &cnode);
  Status InferAsLossDivisorByLayout() override;

  size_t axis_ = 0;
  size_t skip_redistribution_ = false;
  TensorLayout output_infer_tensor_layout_;
};

class SplitVInfo : public SplitInfo {
 public:
  SplitVInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
             const PrimitiveAttrs &attrs)
      : SplitInfo(operator_name, inputs_shape, outputs_shape, attrs) {}
  ~SplitVInfo() override = default;

 protected:
  Status GetAttrs() override;
  Status InferMirrorOps() override { return OperatorInfo::InferMirrorOps(); }
  Status ComputeReplaceGraphForInterleaved(const CNodePtr &cnode) override;
};
class SplitWithSizeInfo : public SplitInfo {
 public:
  SplitWithSizeInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                    const PrimitiveAttrs &attrs)
      : SplitInfo(operator_name, inputs_shape, outputs_shape, attrs) {}
  ~SplitWithSizeInfo() override = default;
  void ReplaceNodeInputOrAttrs() override;

 protected:
  Status GetAttrs() override;
};
class SplitTensorInfo : public SplitInfo {
 public:
  SplitTensorInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                  const PrimitiveAttrs &attrs)
      : SplitInfo(operator_name, inputs_shape, outputs_shape, attrs) {}
  ~SplitTensorInfo() override = default;

 protected:
  Status GetAttrs() override;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SPLIT_INFO_H_
