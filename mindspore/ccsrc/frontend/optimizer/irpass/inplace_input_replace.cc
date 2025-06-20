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

#include "frontend/optimizer/irpass/inplace_input_replace.h"

#include <map>
#include <string>
#include <vector>
#include <utility>
#include "mindspore/ops/op_def/other_ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
bool IsInplaceCNode(const AnfNodePtr &node) {
  if (IsPrimitiveCNode(node, prim::kPrimVirtualViewGrad)) {
    return true;
  }
  auto prim = GetCNodePrimitive(node);
  return prim != nullptr && prim->inplace_prim();
}

bool HasIOMonadInput(const AnfNodePtr &node) {
  if (!irpass::IsCNode(node)) {
    return false;
  }
  const auto &cnode_inputs = node->cast<CNodePtr>()->inputs();
  return std::any_of(cnode_inputs.begin(), cnode_inputs.end(),
                     [](const AnfNodePtr &input) { return IsValueNode<IOMonad>(input); });
}

AnfNodePtr FindNodeUserWithIOMonad(const mindspore::CompactSet<std::pair<AnfNodePtr, int>> &node_users) {
  AnfNodePtr node_user_with_io_monad = nullptr;
  bool found = std::any_of(node_users.begin(), node_users.end(),
                           [&node_user_with_io_monad](const std::pair<AnfNodePtr, int> &node_user) {
                             bool has_io_monad_input = HasIOMonadInput(node_user.first);
                             if (has_io_monad_input) {
                               node_user_with_io_monad = node_user.first;
                             }
                             return has_io_monad_input;
                           });
  return found ? node_user_with_io_monad : nullptr;
}

/**
 * \brief Change inplace input of cnode in func_graph.
 *
 * \example
 * Change from:
 *   %0 = InplaceOp(param_x, param_y)
 *   %1 = UpdataState(U, %0)
 *   %2 = Depend(param_x, %1)
 * To:
 *   %0 = InplaceOp(param_x, param_y)
 *   %1 = UpdataState(U, %0)
 *   %2 = Depend(%0, %1)
 *
 * \param[in] func_graph func graph.
 **/
void ChangeInplaceInputInner(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  std::map<AnfNodePtr, AnfNodePtr> inplace_input;
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto &node_users_map = manager->node_users();
  for (auto node : TopoSort(func_graph->return_node())) {
    if (!irpass::IsCNode(node) || IsPrimitiveCNode(node, prim::kPrimVirtualAssignAdd) ||
        node->func_graph() != func_graph) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    // If cnode has users with io_monad, do not do replacement
    // cnode1 = Load(param, u)
    // Print(str, cnode1, io)
    if (auto it = node_users_map.find(cnode); it != node_users_map.end()) {
      auto &node_users = it->second;
      auto node_user_with_io_monad = FindNodeUserWithIOMonad(node_users);
      if (node_user_with_io_monad != nullptr) {
        MS_LOG(INFO) << "CNode has users with io_monad, no need do replacement, cnode: " << cnode->DebugString()
                     << " , user cnode: " << node_user_with_io_monad->DebugString();
        continue;
      }
    }

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
      manager->SetEdge(cnode, i, repalced_node);
    }
    const auto &prim = GetCNodePrimitive(cnode);
    if (prim == nullptr) {
      continue;
    }
    if (prim->inplace_prim()) {
      const auto &indexes = prim->inplace_input_indexes();
      if (indexes.size() != 1) {
        continue;
      }
      inplace_input[cnode->input(LongToSize(indexes[0] + 1))] = cnode;
      MS_LOG(INFO) << "Record cnode as inplace node: " << cnode->DebugString();
    } else if (IsPrimitiveCNode(node, prim::kPrimVirtualViewGrad)) {
      inplace_input[cnode->input(1)] = cnode;
      MS_LOG(INFO) << "Record VirtualViewGrad cnode as inplace node: " << cnode->DebugString();
    }
  }
  return;
}
}  // namespace

void DoInplaceInputReplace(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  const auto &all_nodes = TopoSort(func_graph->return_node(), SuccDeeperSimple);
  bool exist_inplace_nodes = std::any_of(all_nodes.begin(), all_nodes.end(), IsInplaceCNode);
  if (!exist_inplace_nodes) {
    return;
  }

  // Do inplace input replace for func_graph and sub_graphs
  ChangeInplaceInputInner(func_graph);
  auto sub_graphs = func_graph->func_graphs_used_total();
  for (const auto &sub_graph : sub_graphs) {
    ChangeInplaceInputInner(sub_graph);
  }

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("opt_do_inplace_input_replace.ir", func_graph);
  }
#endif
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
