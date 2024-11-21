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

#include "frontend/parallel/pass/interleave_split_concat_branches.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <set>
#include <string>
#include <queue>
#include <stack>
#include <unordered_map>
#include "mindspore/ops/op_def/array_op_name.h"
#include "mindspore/ops/op_def/math_op_name.h"
#include "mindspore/ops/op_def/other_op_name.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"
#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "frontend/parallel/pass/interleave_branches_utils.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace parallel {
auto const kEnableInterleave = "enable_interleave";

void InterleaveSplitConcatBranches(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_ENABLE_INTERLEAVE_SPLIT_CONCAT_BRANCH);
  if (!is_enable) {
    return;
  }

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<InterLeaveScopePtr> interleave_scopes;

  // Use unique id set to find correspond backward ops
  std::set<std::string> target_unique_id_set;
  size_t scope_id = 0;
  for (const auto &child_graph : manager->func_graphs()) {
    MS_EXCEPTION_IF_NULL(child_graph);
    auto graph_orders = child_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    InterLeaveScopePtr current_scope = nullptr;
    std::stack<InterLeaveScopePtr> scope_stack;

    for (const auto &node : origin_nodes_topological) {
      MS_EXCEPTION_IF_NULL(node);
      EraseInterLeaveBranchAttr(node);
      // Only split and concat node is the target node
      bool is_split = IsPrimitiveCNode(node, prim::kPrimSplit);
      bool is_concat = IsPrimitiveCNode(node, prim::kPrimConcat);
      if (!is_split && !is_concat) {
        continue;
      }

      bool forward = true;
      auto prim = common::AnfAlgo::GetCNodePrimitive(node);
      MS_EXCEPTION_IF_NULL(prim);
      // Node with enable interleave flag or its backward node
      if (!prim->HasAttr(kEnableInterleave)) {
        if (!node->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
          continue;
        }
        if (target_unique_id_set.find(GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrForwardUniqueId))) ==
            target_unique_id_set.end()) {
          continue;
        }
        forward = false;
      }

      if (forward && node->HasPrimalAttr(kPrimalAttrUniqueId)) {
        (void)target_unique_id_set.emplace(GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrUniqueId)));
      }

      if (is_split) {
        current_scope = std::make_shared<InterLeaveScope>();
        current_scope->graph = child_graph;
        current_scope->fork_node = node;
        current_scope->forward = forward;
        current_scope->scope_id = ++scope_id;
        scope_stack.push(current_scope);
      }

      if (is_concat) {
        if (scope_stack.empty()) {
          continue;
        }
        current_scope = scope_stack.top();
        current_scope->merge_node = node;
        interleave_scopes.emplace_back(current_scope);
        scope_stack.pop();
      }
    }
  }

  // Handle each split/concat scope
  for (auto &interleave_scope : interleave_scopes) {
    InterleaveParallelBranches(interleave_scope);
  }
}
}  // namespace parallel
}  // namespace mindspore
