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

#ifndef MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INPLACE_INPUT_REPLACE_H_
#define MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INPLACE_INPUT_REPLACE_H_

#include <map>

#include "frontend/optimizer/optimizer.h"
#include "frontend/optimizer/irpass.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
class InplaceInputReplace {
 public:
  bool operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
    bool change = ChangeInplaceInputInner(func_graph);
    const auto &sub_graphs = func_graph->func_graphs_used_total();
    for (auto sub_graph : sub_graphs) {
      change = ChangeInplaceInputInner(sub_graph) || change;
    }
    return change;
  }

 private:
  // Change from:
  // %0 = InplaceOp(param_x, param_y)
  // %1 = UpdataState(U, %0)
  // %2 = Depend(param_x, %1)
  // To:
  // %0 = InplaceOp(param_x, param_y)
  // %1 = UpdataState(U, %0)
  // %2 = Depend(%0, %1)
  bool ChangeInplaceInputInner(const FuncGraphPtr &func_graph) {
    MS_EXCEPTION_IF_NULL(func_graph);
    std::map<AnfNodePtr, AnfNodePtr> inplace_input;
    bool change = false;
    for (auto node : TopoSort(func_graph->return_node())) {
      if (!IsCNode(node) || IsPrimitiveCNode(node, prim::kPrimVirtualAssignAdd) || node->func_graph() != func_graph) {
        continue;
      }
      auto cnode = node->cast<CNodePtr>();

      for (size_t i = 1; i < cnode->size(); i++) {
        auto original_input = cnode->input(i);
        if (inplace_input.count(original_input) == 0 || original_input->func_graph() != func_graph) {
          continue;
        }
        // Find the final inplaced cnode to replace
        // For example:
        // %1 = Inplace(%0)
        // %2 = Inplace(%1)
        // %3 = Depend(%0, U) ==> %3 = Depend(%2, U)
        AnfNodePtr repalced_node = inplace_input[original_input];
        while (inplace_input.count(repalced_node) != 0) {
          repalced_node = inplace_input[repalced_node];
        }
        MS_LOG(INFO) << "Replace cnode : " << cnode->DebugString() << " input from: " << original_input->DebugString()
                     << " to: " << repalced_node->DebugString() << " for inplace ops replacement.";
        cnode->set_input(i, repalced_node);
        change = true;
      }
      const auto &prim = GetCNodePrimitive(cnode);
      if (prim != nullptr && prim->inplace_prim() && prim->rw_write_input_indexes().size() == 1) {
        size_t index = prim->rw_write_input_indexes()[0];
        inplace_input[cnode->input(index + 1)] = cnode;
        MS_LOG(INFO) << "Record cnode as inplace node: " << cnode->DebugString();
      }
    }
    return change;
  }
};
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_FRONTEND_OPTIMIZER_IRPASS_INPLACE_INPUT_REPLACE_H_
