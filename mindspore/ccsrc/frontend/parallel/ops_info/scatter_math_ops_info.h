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

#ifndef MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SCATTER_MATH_OPS_INFO_H_
#define MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SCATTER_MATH_OPS_INFO_H_

#include <string>
#include <memory>
#include <vector>

#include "utils/hash_map.h"
#include "ir/value.h"
#include "frontend/parallel/auto_parallel/operator_costmodel.h"
#include "frontend/parallel/ops_info/scatter_ops_info.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "frontend/parallel/strategy.h"

namespace mindspore {
namespace parallel {
class ScatterMathOpsInfo : public ScatterOpsInfo {
 public:
  ScatterMathOpsInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                     const PrimitiveAttrs &attrs, const OperatorCostPtr &cost)
      : ScatterOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, cost) {}
  ~ScatterMathOpsInfo() override = default;
  std::vector<StrategyPtr> GenerateOpStrategies(int64_t stage_id) override;
  ReplaceGraphPtr replace_graph(const CNodePtr &cnode) override;
  Status InitForCostModel(const StrategyPtr &in_strategy, const StrategyPtr &out_strategy) override;

 protected:
  Status GetAttrs() override { return SUCCESS; }
  Status CheckStrategy(const StrategyPtr &strategy) override;
  Status InferMirrorOps() override { return SUCCESS; }  // the scatter_update only use in eval/predict
  Status InferForwardCommunication() override { return SUCCESS; }
  Status InferTensorMap() override;
  virtual Status ComputeReplaceGraph(const CNodePtr &cnode);
  Status InferBias();
  Status CheckStrategyForDynamicShape(const StrategyPtr &strategy) override;
  bool do_replace_graph_ = false;
  int64_t bias_ = 0;
  int64_t slice_size_ = 0;
};

class ScatterAddInfo : public ScatterMathOpsInfo {
 public:
  ScatterAddInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ScatterMathOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<ScatterMathOpsCost>()) {}
  ~ScatterAddInfo() override = default;

 protected:
  Status ComputeReplaceGraph(const CNodePtr &cnode) override;
};

class ScatterSubInfo : public ScatterAddInfo {
 public:
  ScatterSubInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ScatterAddInfo(operator_name, inputs_shape, outputs_shape, attrs) {}
  ~ScatterSubInfo() override = default;
};

class ScatterMulInfo : public ScatterMathOpsInfo {
 public:
  ScatterMulInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ScatterMathOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<ScatterMathOpsCost>()) {}
  ~ScatterMulInfo() override = default;
};

class ScatterDivInfo : public ScatterMathOpsInfo {
 public:
  ScatterDivInfo(const std::string &operator_name, const Shapes &inputs_shape, const Shapes &outputs_shape,
                 const PrimitiveAttrs &attrs)
      : ScatterMathOpsInfo(operator_name, inputs_shape, outputs_shape, attrs, std::make_shared<ScatterMathOpsCost>()) {}
  ~ScatterDivInfo() override = default;
};

using ScatterAddInfoPtr = std::shared_ptr<ScatterAddInfo>;
using ScatterSubInfoPtr = std::shared_ptr<ScatterSubInfo>;
using ScatterMulInfoPtr = std::shared_ptr<ScatterMulInfo>;
using ScatterDivInfoPtr = std::shared_ptr<ScatterDivInfo>;
}  // namespace parallel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_PARALLEL_OPS_INFO_SCATTER_MATH_OPS_INFO_H_
