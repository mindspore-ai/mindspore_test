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

#include "frontend/parallel/graph_util/parallel_tensordump.h"

#include <vector>
#include <memory>
#include <utility>
#include <string>
#include "ir/anf.h"
#include "ir/manager.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "mindspore/ops/op_def/structure_ops.h"

namespace mindspore {
namespace parallel {
ParallelTensorDumpHandler::ParallelTensorDumpHandler(
  const std::vector<std::pair<std::pair<AnfNodePtr, int>, std::vector<int>>> &next_nodes) {
  nodes_need_redistribution_ = next_nodes;
  for (auto &node_info : nodes_need_redistribution_) {
    auto node = node_info.first.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    size_t pos = IntToSize(node_info.first.second);
    if (pos >= node->inputs().size()) {
      MS_LOG(ERROR) << "Access Index should less than length of node input."
                    << " Length of node input: " << node->inputs().size() << " Access Index: " << pos;
    }
    AnfNodePtr parent = node->input(pos);
    MS_EXCEPTION_IF_NULL(parent);
    parent_to_successors_[parent].push_back(node_info.first);
  }
}

void ParallelTensorDumpHandler::HandleParallelTensorDump() {
  for (auto &parent_successors_pair : parent_to_successors_) {
    const AnfNodePtr &parent = parent_successors_pair.first;
    std::vector<std::pair<AnfNodePtr, int>> &successors = parent_successors_pair.second;
    MS_EXCEPTION_IF_NULL(parent);
    FuncGraphPtr func_graph = parent->func_graph();
    MS_EXCEPTION_IF_NULL(func_graph);
    FuncGraphManagerPtr manager = func_graph->manager();
    MS_EXCEPTION_IF_NULL(manager);
    AnfNodePtrList dumps = CollectSuccessorDumpNodes(parent, manager);
    for (auto &node_pair : successors) {
      AnfNodePtr node = node_pair.first;
      CNodePtr cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      size_t index = IntToSize(node_pair.second);
      if (parent == cnode->input(index)) {
        MS_LOG(INFO) << "No Redistribution Operators Insert.";
      }
      AnfNodePtr last_inserted_redistribution_op = cnode->input(index);
      MS_EXCEPTION_IF_NULL(last_inserted_redistribution_op);
      MS_LOG(WARNING) << "Last Insert Redistribution: " << last_inserted_redistribution_op->DebugString();
      (void)ProcessTensorDumps(dumps, cnode, index, last_inserted_redistribution_op, func_graph, cnode->scope());
    }
    for (auto &dump_node : dumps) {
      auto prim = GetCNodePrimitive(dump_node);
      MS_EXCEPTION_IF_NULL(prim);
      ValuePtr attr_input_output = prim->GetAttr("input_output");
      std::string str_input_output = GetValue<std::string>(attr_input_output);
      if (str_input_output == "in") {
        (void)manager->Replace(dump_node, parent);
      }
    }
  }
}

AnfNodePtrList ParallelTensorDumpHandler::CollectSuccessorDumpNodes(const AnfNodePtr &parent_of_dump_nodes,
                                                                    const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(parent_of_dump_nodes);
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodePtrList dumps;
  AnfNodeIndexSet node_users = manager->node_users()[parent_of_dump_nodes];
  MS_LOG(INFO) << "Node Parent is: " << parent_of_dump_nodes->DebugString();
  // search tensordump op in successors of parameter parent_of_dump_nodes
  for (auto &node_pair : node_users) {
    AnfNodePtr anf_node = node_pair.first;
    CNodePtr cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    MS_LOG(INFO) << "Parent node's successor: " << cnode->DebugString();
    if (IsPrimitiveCNode(cnode, prim::kPrimTensorDump)) {
      dumps.push_back(anf_node);
    }
  }
  return dumps;
}

void ParallelTensorDumpHandler::InsertNewTensorDump(const CNodePtr &dump_cnode,
                                                    const AnfNodePtr &last_insert_redistribution_op,
                                                    const CNodePtr &node, const size_t pos_u,
                                                    const FuncGraphPtr &func_graph, const ScopePtr &scope,
                                                    const std::string &dump_mode) {
  // Create New TensorDump Node to display the input of node
  MS_EXCEPTION_IF_NULL(func_graph);
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(dump_cnode->scope());
  MS_EXCEPTION_IF_NULL(node->scope());
  if (node->scope()->name().find(dump_cnode->scope()->name()) == std::string::npos) {
    // If TensorDump node's scope is not prefix of node scope,
    // Don't insert new TensorDump Node as successor of last_insert_redistribution_op.
    return;
  }
  ValuePtr v = GetValueNode(dump_cnode->input(1));
  auto dump_cnode_filepath = GetValue<std::string>(v);
  size_t p = dump_cnode_filepath.rfind(".npy");
  if (p != std::string::npos) {
    dump_cnode_filepath.erase(p);
  }
  std::string name = dump_cnode_filepath + "_" + dump_mode + ".npy";
  ValueNodePtr name_value = NewValueNode(name);
  auto new_dump_node = func_graph->NewCNode(
    {NewValueNode(prim::kPrimTensorDump->Clone()), name_value, last_insert_redistribution_op, NewValueNode(kIOMonad)});
  // ops of update_state and depend are used to keep sequence
  auto monad_node = NewValueNode(kIOMonad);
  auto new_update_state_node = func_graph->NewCNode({NewValueNode(prim::kPrimUpdateState), monad_node, new_dump_node});
  auto depend_prev = node->input(pos_u);
  auto new_depend_node = func_graph->NewCNode({NewValueNode(prim::kPrimDepend), depend_prev, new_update_state_node});
  // config new nodes attributes
  PrimitivePtr new_dump_prim = GetCNodePrimitive(new_dump_node);
  MS_EXCEPTION_IF_NULL(new_dump_prim);
  new_dump_prim->set_instance_name(name + "_new_generate");
  new_dump_prim->AddAttr("side_effect_io", MakeValue(true));
  new_dump_prim->AddAttr("input_output", MakeValue(dump_mode + "_inserted"));
  new_dump_prim->AddAttr("channel_name", MakeValue("ms_tensor_dump"));
  // Avoid CSE optimization
  new_dump_prim->AddAttr("side_effect_hidden", MakeValue(true));
  new_dump_node->set_scope(scope);
  new_dump_node->input(0)->set_scope(scope);
  MS_LOG(INFO) << "New Dump Node: " << new_dump_node->DebugString();
  (void)manager->SetEdge(node, pos_u, new_depend_node);
}

void ParallelTensorDumpHandler::ProcessTensorDumps(const std::vector<AnfNodePtr> &dumps, const CNodePtr &node,
                                                   const size_t pos_u, const AnfNodePtr &last_insert_op,
                                                   const FuncGraphPtr &func_graph, const ScopePtr &scope) {
  FuncGraphManagerPtr manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(scope);
  for (auto &anf_node : dumps) {
    CNodePtr dump_cnode = anf_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(dump_cnode);
    MS_LOG(INFO) << "Current Process Dump Node: " << dump_cnode->DebugString();
    PrimitivePtr prim = GetCNodePrimitive(dump_cnode);
    MS_EXCEPTION_IF_NULL(prim);
    std::string dump_mode = GetValue<std::string>(prim->GetAttr("input_output"));
    if (dump_mode == "out") {
      // Default setting do nothing
    } else if (dump_mode == "all") {
      MS_LOG(ERROR) << "TensorDump's parameter mode has deprecated 'all' value."
                    << "Now parameter mode only support value in [out, in].";
    } else if (dump_mode == "in") {
      InsertNewTensorDump(dump_cnode, last_insert_op, node, pos_u, func_graph, scope, "in");
    } else {
      MS_LOG(ERROR) << "Dump mode of " << dump_mode << "is not supported, only support mode in [out, in]";
    }
  }
}

}  // namespace parallel
}  // namespace mindspore
