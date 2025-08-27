/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/shard_eliminate.h"
#include "ir/func_graph_flag.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace internal {
AnfNodePtr ExpandShard(const CNodePtr &node, bool preserve_defer_inline = false) {
  auto vnode = node->input(1)->cast<ValueNodePtr>();
  auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
  MS_EXCEPTION_IF_NULL(func_graph);
  func_graph->erase_flag(FUNC_GRAPH_FLAG_DEFER_INLINE);
  if (preserve_defer_inline) {
    func_graph->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
  }
  return NewValueNode(func_graph);
}

AnfNodePtr ExpandAddAttr(const CNodePtr &node) {
  auto vnode = node->input(kIndex1)->cast<ValueNodePtr>();
  auto func_graph = GetValueNode<FuncGraphPtr>(vnode);
  MS_EXCEPTION_IF_NULL(func_graph);
  return NewValueNode(func_graph);
}
}  // namespace internal

bool ExpandShardPrim::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  bool change = false;
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &shard_node : prim_nodes_) {
    auto shard_node_fg = shard_node->func_graph();
    // Preserve 'defer_inline' flag for shard node in the root graph to avoid premature inlining.
    bool preserve_defer_inline = (shard_node_fg == func_graph);
    auto expanded_shard = internal::ExpandShard(shard_node, preserve_defer_inline);

    if (preserve_defer_inline) {
      for (const auto &node : func_graph->nodes()) {
        auto cnode = node->cast<CNodePtr>();
        if (cnode == nullptr) {
          continue;
        }
        if (IsPrimitiveCNode(cnode, prim::kPrimJ)) {
          manager->SetEdge(cnode, kIndex1, expanded_shard);
        }
      }
    }

    (void)manager->Replace(shard_node, expanded_shard);
    change = true;
  }
  return change;
}

bool ExpandAddAttrPrim::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  bool change = false;
  auto manager = optimizer->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto &node : prim_nodes_) {
    if (!IsPrimitiveCNode(node, prim::kPrimAddAttr)) {
      // addattr pass before shard, skip shard.
      continue;
    }
    auto expanded_addattr_node = internal::ExpandAddAttr(node);
    (void)manager->Replace(node, expanded_addattr_node);
    change = true;
  }
  return change;
}

}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
