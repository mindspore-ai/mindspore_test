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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RMS_NORM_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RMS_NORM_INFO_H_

#include <string>
#include <memory>
#include <vector>
#include <utility>

#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/graph_util/generate_graph.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
constexpr size_t RMS_NORM_INPUT_SIZE = 2;
constexpr size_t RMS_NORM_INPUT_INDEX = 0;
constexpr size_t RMS_NORM_GAMMA_INDEX = 1;
constexpr float DEFAULT_EPS = 1e-6;
constexpr char BEGIN_NORM_AXIS[] = "begin_norm_axis";

class RmsNormInfo : public OperatorInfo {
 public:
  RmsNormInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
              const PrimitiveAttrs &attrs)
      : OperatorInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<LayerNormCost>()),
        begin_norm_axis_(0) {}
  ~RmsNormInfo() override = default;

  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  Status SetCostUnderStrategy(const StrategyPtr &strategy) override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;

 protected:
  Status GetAttrs() override;
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferDevMatrixShape() override;
  Status InferTensorMap() override;
  Status InferAsLossDivisor() override;
  Status InferAsLossDivisorByLayout() override;
  Status CreateInputTensorMap(size_t input_index);
  Status GenerateGammaStrategies(const std::vector<StrategyPtr> &sp_vector);
  Status InitShapes();
  Status InferOutputTensorInfo() override;
  Status CheckInputLayout() override;
  Status CheckOutputLayout() override;
  Status InferOutputLayout();
  std::string CreateCommGroupFromRankList(const RankList &rank_list);
  RankList GetAllReduceRankList();
  AnfNodePtr GetInputOutputNodeForSplitNormAxis(const CNodePtr &cnode, const AnfNodePtr &square_actual_input_node,
                                                GenerateGraph *gen_g,
                                                std::vector<std::pair<AnfNodePtr, int64_t>> *input_nodes);
  Status ComputeReplaceGraphForInterleaved(const CNodePtr &cnode);
  Status ComputeReplaceGraphForSplitNormAxis(const CNodePtr &cnode);

 private:
  size_t begin_norm_axis_;
  bool norm_axis_splitted_ = false;
  TensorLayout output_infer_tensor_layout_;
  TensorLayout rstd_infer_tensor_layout_;
  Shape input_shape_;
  Shape gamma_shape_;
};
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_RMS_NORM_INFO_H_
