/**
 * Copyright 2024-2025Huawei Technologies Co., Ltd
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
#include <unordered_set>
#include <algorithm>
#include <memory>
#include <utility>
#include <string>
#include "ir/anf.h"
#include "base/base.h"
#include "ir/manager.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/ops_info/operator_info.h"
#include "mindspore/ops/op_def/structure_ops.h"

namespace mindspore {
namespace parallel {
ParallelTensorDumpHandler::ParallelTensorDumpHandler(
  const std::vector<AnfNodePtr> &pre_nodes,
  const std::vector<std::pair<std::pair<AnfNodePtr, int>, std::vector<int>>> &next_nodes) {
  // vector pre_nodes size is 1
  prenode_redistribution_ = pre_nodes.front();
  nodes_need_redistribution_ = next_nodes;
  for (auto &node_info : nodes_need_redistribution_) {
    auto node = node_info.first.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(node);
    size_t pos = IntToSize(node_info.first.second);
    if (pos >= node->inputs().size()) {
      MS_LOG(ERROR) << "Access Index should less than length of node input."
                    << " Length of node input: " << node->inputs().size() << " Access Index: " << pos;
    }
    MS_EXCEPTION_IF_NULL(node->input(pos));
    parent_to_successors_[prenode_redistribution_].push_back(node_info.first);
  }
}

void ParallelTensorDumpHandler::HandleParallelTensorDump() {
  for (auto &parent_successors_pair : parent_to_successors_) {
    const AnfNodePtr &parent = parent_successors_pair.first;
    std::vector<std::pair<AnfNodePtr, int>> &successors = parent_successors_pair.second;
    MS_EXCEPTION_IF_NULL(parent);
    std::unordered_set<AnfNodePtr> viewed_dumps;
    for (auto &node_pair : successors) {
      AnfNodePtr node = node_pair.first;
      MS_EXCEPTION_IF_NULL(node);
      FuncGraphPtr func_graph = node->func_graph();
      MS_EXCEPTION_IF_NULL(func_graph);
      FuncGraphManagerPtr manager = func_graph->manager();
      MS_EXCEPTION_IF_NULL(manager);
      AnfNodePtrList path = CollectNodePathBetween(parent, node_pair);
      AnfNodePtrList dumps = CollectDumpNodesAlongPath(path, manager);
      viewed_dumps.insert(dumps.begin(), dumps.end());
      CNodePtr cnode = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      size_t index = IntToSize(node_pair.second);
      if (parent == cnode->input(index)) {
        MS_LOG(INFO) << "No Redistribution Operators Insert.";
      }
      AnfNodePtr last_inserted_redistribution_op = cnode->input(index);
      MS_EXCEPTION_IF_NULL(last_inserted_redistribution_op);
      MS_LOG(INFO) << "Last Insert Redistribution: " << last_inserted_redistribution_op->DebugString();
      (void)ProcessTensorDumps(dumps, cnode, index, last_inserted_redistribution_op, func_graph, cnode->scope());
    }
    for (auto &dump_node : viewed_dumps) {
      FuncGraphPtr fg = dump_node->func_graph();
      MS_EXCEPTION_IF_NULL(fg);
      FuncGraphManagerPtr fg_mng = fg->manager();
      MS_EXCEPTION_IF_NULL(fg_mng);
      CNodePtr dump_cnode = dump_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(dump_cnode);
      AnfNodePtr dump_node_parent = dump_cnode->input(kIndex2);
      MS_EXCEPTION_IF_NULL(dump_node_parent);
      auto prim = GetCNodePrimitive(dump_node);
      MS_EXCEPTION_IF_NULL(prim);
      ValuePtr attr_input_output = prim->GetAttr("input_output");
      std::string str_input_output = GetValue<std::string>(attr_input_output);
      if (str_input_output != "in") {
        continue;
      }
      ValueNodePtr dump_path_value_node = dump_cnode->input(kIndex1)->cast<ValueNodePtr>();
      ValuePtr v = GetValueNode(dump_path_value_node);
      std::string dump_cnode_filepath = GetValue<std::string>(v);
      if (tensordump_need_remove_.count(dump_cnode) != 0) {
        MS_LOG(DEBUG) << dump_cnode->DebugString() << " will be removed from graph";
        (void)fg_mng->Replace(dump_node, dump_node_parent);
      } else {
        MS_LOG(DEBUG) << dump_cnode->DebugString() << " will not be removed from graph";
        size_t p = dump_cnode_filepath.rfind(".npy");
        if (p != std::string::npos) {
          dump_cnode_filepath.erase(p);
        }
        std::string name = dump_cnode_filepath + "_in.npy";
        fg_mng->SetEdge(dump_cnode, kIndex1, NewValueNode(name));
      }
    }
    tensordump_need_remove_.clear();
  }
}

AnfNodePtrList ParallelTensorDumpHandler::CollectNodePathBetween(AnfNodePtr start, std::pair<AnfNodePtr, int> end) {
  MS_EXCEPTION_IF_NULL(start);
  AnfNodePtrList path;
  MS_EXCEPTION_IF_NULL(end.first);
  path.push_back(end.first);
  CNodePtr end_cnode = end.first->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(end_cnode);
  AnfNodePtr cur_node = end_cnode->input(IntToSize(end.second));
  MS_EXCEPTION_IF_NULL(cur_node);
  path.push_back(cur_node);
  if (!cur_node->isa<CNode>()) {
    return path;
  }
  CNodePtr cur_cnode = cur_node->cast<CNodePtr>();
  while (GetCNodePrimitive(cur_cnode) &&
         GetCNodePrimitive(cur_cnode)->instance_name().find("redistribution_op") != std::string::npos) {
    auto cnode_parent = cur_cnode->input(kIndex1);
    path.push_back(cnode_parent);
    cur_cnode = cnode_parent->cast<CNodePtr>();
  }
  while (IsSomePrimitiveList(cur_cnode, {DEPEND, INSERTGRADIENTOF, CAST})) {
    cur_node = cur_cnode->input(kIndex1);
    path.push_back(cur_node);
    cur_cnode = cur_node->cast<CNodePtr>();
  }
  return path;
}

AnfNodePtrList ParallelTensorDumpHandler::CollectDumpNodesAlongPath(const AnfNodePtrList &path,
                                                                    const FuncGraphManagerPtr &manager) {
  AnfNodePtrList dumps;
  for (size_t i = 1; i < path.size(); i++) {
    MS_EXCEPTION_IF_NULL(path[i]);
    const AnfNodePtrList &local_dumps = CollectSuccessorDumpNodes(path[i], manager);
    std::copy(local_dumps.begin(), local_dumps.end(), std::back_inserter(dumps));
  }
  return dumps;
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
  bool is_side_effect_tensordump = dump_cnode->inputs().size() == 4 ? true : false;
  if (node->scope()->name() != dump_cnode->scope()->name()) {
    // If TensorDump node's scope is not same with node scope,
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
  tensordump_need_remove_.insert(dump_cnode);
  CNodePtr new_dump_node;
  if (is_side_effect_tensordump) {
    new_dump_node = func_graph->NewCNode({NewValueNode(prim::kPrimTensorDump->Clone()), name_value,
                                          last_insert_redistribution_op, NewValueNode(kIOMonad)});
  } else {
    new_dump_node =
      func_graph->NewCNode({NewValueNode(prim::kPrimTensorDump->Clone()), name_value, last_insert_redistribution_op});
  }
  // ops of update_state and depend are used to keep sequence
  auto monad_node = NewValueNode(kIOMonad);
  auto new_update_state_node = func_graph->NewCNode({NewValueNode(prim::kPrimUpdateState), monad_node, new_dump_node});
  auto depend_prev = node->input(pos_u);
  auto new_depend_kIndex2_input = is_side_effect_tensordump ? new_update_state_node : new_dump_node;
  auto new_depend_node = func_graph->NewCNode({NewValueNode(prim::kPrimDepend), depend_prev, new_depend_kIndex2_input});
  // config new nodes attributes
  PrimitivePtr new_dump_prim = GetCNodePrimitive(new_dump_node);
  MS_EXCEPTION_IF_NULL(new_dump_prim);
  new_dump_prim->set_instance_name(name + "_new_generate");
  new_dump_prim->AddAttr("side_effect_io", MakeValue(is_side_effect_tensordump));
  new_dump_prim->AddAttr("input_output", MakeValue(dump_mode + "_inserted"));
  new_dump_prim->AddAttr("channel_name", MakeValue("ms_tensor_dump"));
  // Avoid CSE optimization
  new_dump_prim->AddAttr("side_effect_hidden", MakeValue(true));
  if (!is_side_effect_tensordump) {
    new_dump_prim->AddAttr("dyn_input_sizes", MakeValue(std::vector<int>{-1, 1}));
  }
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
