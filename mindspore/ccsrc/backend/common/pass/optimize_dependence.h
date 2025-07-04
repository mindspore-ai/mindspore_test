/**
 * Copyright 2019 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_OPTIMIZE_DEPENDENCE_H_
#define MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_OPTIMIZE_DEPENDENCE_H_

#include "include/backend/optimizer/optimizer.h"

namespace mindspore {
namespace opt {
class BACKEND_COMMON_EXPORT OptimizeDependence : public PatternProcessPass {
 public:
  explicit OptimizeDependence(bool multigraph = true) : PatternProcessPass("optimize_dependence", multigraph) {}
  ~OptimizeDependence() override = default;
  const BaseRef DefinePattern() const override;
  const AnfNodePtr Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const EquivPtr &) const override;
  const AnfNodePtr GetConvertNode(const FuncGraphPtr &graph, const AnfNodePtr &node, const size_t index) const;
};
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_OPTIMIZER_PASS_OPTIMIZE_DEPENDENCE_H_
