/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_GRAD_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_GRAD_H_

#include <vector>
#include "ir/anf.h"
#include "ir/meta_func_graph.h"
#include "frontend/optimizer/optimizer.h"

namespace mindspore {
namespace ad {
// We pass the bprop_auto_monad level to the ad::Grad, and then keep the order between forward and backward in the later
// pass 'add_forward_monad_depend'.
enum BpropAutoMonadLevel : int {
  // When setting to None level, it will not keep the order for all side effect nodes between forward and backward.
  kLevelNone = 0,
  // When setting to Top level, ir will keep the order for all side effect nodes between forward inputs and backward.
  kLevelTop,
  // When setting to Whole level, it will keep the order for all side effect nodes between forward and backward.
  kLevelWhole,
};
FRONTEND_EXPORT FuncGraphPtr Grad(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer,
                                  bool is_top = true, BpropAutoMonadLevel level = kLevelNone,
                                  bool is_view_inplace = false, bool is_grad_by_j = false);
FuncGraphVector GradMultiFuncGraph(const FuncGraphVector &func_graphs, const opt::OptimizerPtr &optimizer,
                                   const std::vector<bool> &is_view_inplace, bool is_top = true,
                                   bool is_grad_by_j = false);
FuncGraphPtr Kprim(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources);
MetaFuncGraphPtr Kmeta(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &, const AnfNodePtr &node);
FRONTEND_EXPORT void CleanRes();
bool MergeForward(const FuncGraphPtr &root, const opt::OptimizerPtr &opt);
}  // namespace ad
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_AD_GRAD_H_
