/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GATHER_PRE_RMS_NORM_FUSION_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GATHER_PRE_RMS_NORM_FUSION_H_
#include <vector>
#include <string>
#include <memory>
#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class GatherPreRmsNormFusion : public PatternProcessPass {
 public:
  explicit GatherPreRmsNormFusion(bool multigraph = true)
      : PatternProcessPass("gather_pre_rms_norm_fusion", multigraph) {
    x_ = std::make_shared<Var>();
    res_in_ = std::make_shared<Var>();
    indices_ = std::make_shared<Var>();
    gamma_ = std::make_shared<Var>();
    epsilon_ = std::make_shared<Var>();
  }
  ~GatherPreRmsNormFusion() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &, const AnfNodePtr &, const EquivPtr &) const override;

 private:
  CNodePtr CreateGatherPreRmsNormNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                      const EquivPtr &equiv) const;
  std::vector<std::string> MustExistPrimitiveName() const override;
  VarPtr x_;
  VarPtr res_in_;
  VarPtr indices_;
  VarPtr gamma_;
  VarPtr epsilon_;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_GATHER_PRE_RMS_NORM_FUSION_H_
