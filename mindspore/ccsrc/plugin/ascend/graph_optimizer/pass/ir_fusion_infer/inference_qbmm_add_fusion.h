/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_QBMM_ADD_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_QBMM_ADD_FUSION_H_

#include "plugin/ascend/graph_optimizer/pass/ir_fusion_infer/inference_qbmm_fusion_base.h"
#include <vector>
#include <string>

namespace mindspore {
namespace opt {
class QbmmAddFusion : public QbmmFusionBase {
 public:
  explicit QbmmAddFusion(const std::string &name = "inference_qbmm_add_fusion", bool multigraph = true)
      : QbmmFusionBase(name, multigraph) {}

  ~QbmmAddFusion() override = default;

  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;
  const BaseRef DefinePattern() const override;

 private:
  CNodePtr CreateQbmmAddNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &equiv) const;
  std::vector<std::string> MustExistPrimitiveName() const override;
};
}  // namespace opt
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_ASCEND_IR_FUSION_INFERENCE_QBMM_ADD_FUSION_H_
