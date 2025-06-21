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

#include "pipeline/jit/ps/remote_memory.h"

#include <utility>
#include <memory>
#include <vector>
#include <set>
#include "ir/core_ops_primitive.h"

namespace mindspore {
namespace remote_memory {
namespace {
struct DetachInfo {
  DetachInfo(AnfNodePtr user_node, AnfNodePtr depend_detach_node, const AnfNodePtrList &before_detach_nodes)
      : user_node_(user_node),
        depend_detach_node_(depend_detach_node),
        before_detach_nodes_(std::move(before_detach_nodes)) {}
  AnfNodePtr user_node_;
  AnfNodePtr depend_detach_node_;
  AnfNodePtrList before_detach_nodes_;
};

struct NodeUserCompare {
  bool operator()(const std::pair<size_t, AnfNodePtr> &lhs, const std::pair<size_t, AnfNodePtr> &rhs) const {
    return lhs.first < rhs.first;
  }
};

using NodeUserCompareList = std::set<std::pair<size_t, AnfNodePtr>, NodeUserCompare>;

NodeUserCompareList CollectOrderedNodeUsers(const FuncGraphManagerPtr &mng, const AnfNodePtrList &topo_orders,
                                            const AnfNodePtr &node) {
  NodeUserCompareList ret;
  for (const auto &user_node : mng->node_users()[node]) {
    auto cur_iter = std::find(topo_orders.begin(), topo_orders.end(), user_node.first);
    size_t cur_index = std::distance(topo_orders.begin(), cur_iter);
    (void)ret.emplace(std::pair<size_t, AnfNodePtr>(cur_index, *cur_iter));
  }
  return ret;
}

std::vector<DetachInfo> CollectDetachInfo(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph,
                                          const AnfNodePtrList &all_nodes) {
  MS_EXCEPTION_IF_NULL(mng);
  std::vector<DetachInfo> detach_info_list;
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimToRemote)) {
      continue;
    }
    const auto &node_users = mng->node_users()[node];
    if (node_users.size() != 1) {
      MS_LOG(EXCEPTION) << "ToRemote node " << node->DebugString() << " should only have one user node.";
    }
    AnfNodePtr user_node = node_users.front().first;
    if (IsPrimitiveCNode(user_node, prim::kPrimDetach)) {
      MS_LOG(INFO) << "Detach node is already added for node " << node->DebugString();
      continue;
    }
    if (!IsPrimitiveCNode(user_node, prim::kPrimDepend)) {
      MS_LOG(EXCEPTION) << "Unexpected user node: " << user_node->DebugString();
    }

    const auto &candidate_nodes_infos = CollectOrderedNodeUsers(mng, all_nodes, user_node);
    AnfNodePtrList nodes_before_detach{NewValueNode(prim::kPrimMakeTuple)};
    for (const auto &pair : candidate_nodes_infos) {
      auto cur_node = pair.second;
      if (IsPrimitiveCNode(cur_node, prim::kPrimPrefetch)) {
        break;
      }
      (void)nodes_before_detach.emplace_back(cur_node);
    }
    if (nodes_before_detach.size() == 1) {
      MS_LOG(INFO) << "No need to add detach for node " << node->DebugString();
      continue;
    }
    const auto &users_for_nodes_before_detach = CollectOrderedNodeUsers(mng, all_nodes, nodes_before_detach.back());
    auto node_to_depend_detach = (*users_for_nodes_before_detach.begin()).second;
    (void)detach_info_list.emplace_back(DetachInfo{user_node, node_to_depend_detach, nodes_before_detach});
  }
  return detach_info_list;
}
}  // namespace

void AddDetachToGraph(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph) {
  const auto &all_nodes = TopoSort(func_graph->output(), SuccDeeperSimple);
  auto tr = mng->Transact();
  const auto &detach_info_list = CollectDetachInfo(mng, func_graph, all_nodes);
  // todo: need to handle multi-fg scene.
  for (const auto &detach_info : detach_info_list) {
    const auto &cur_before_detach_nodes = detach_info.before_detach_nodes_;
    auto tuple_node_before_detach = func_graph->NewCNode(cur_before_detach_nodes);
    abstract::AbstractBasePtrList tuple_elements_abs;
    for (size_t i = 1; i < cur_before_detach_nodes.size(); ++i) {
      MS_EXCEPTION_IF_NULL(cur_before_detach_nodes[i]->abstract());
      (void)tuple_elements_abs.emplace_back(cur_before_detach_nodes[i]->abstract());
    }
    auto tuple_abs = std::make_shared<abstract::AbstractTuple>(tuple_elements_abs);
    tuple_node_before_detach->set_abstract(tuple_abs);
    auto user_node = detach_info.user_node_;
    auto detach_node = func_graph->NewCNode({NewValueNode(prim::kPrimDetach), user_node, tuple_node_before_detach});
    auto detach_node_abs = std::make_shared<abstract::AbstractScalar>(kValueAny, kBool);
    detach_node->set_abstract(detach_node_abs);
    auto node_to_depend_detach = detach_info.depend_detach_node_;
    auto node_after_depend_detach =
      func_graph->NewCNode({NewValueNode(prim::kPrimDepend), node_to_depend_detach, detach_node});
    node_after_depend_detach->set_abstract(node_to_depend_detach->abstract());
    tr.Replace(node_to_depend_detach, node_after_depend_detach);
  }
  tr.Commit();
}

CNodePtr ActivationToRemote(const FuncGraphManagerPtr &mng, const FuncGraphPtr &fprop, const FuncGraphPtr &bprop,
                            const AnfNodePtr &out, const AnfNodePtr &dout, const AnfNodePtr &out_param) {
  // %out = forward(xxx)
  // -->
  // %to_remote = ToRemote(%out)
  // %new_out_value = Depend(%out, %to_remote)
  AnfNodePtrList out_to_remote_inputs{NewValueNode(prim::kPrimToRemote), out};
  auto out_to_remote = fprop->NewCNode(out_to_remote_inputs);
  AnfNodePtrList new_out_value_input{NewValueNode(prim::kPrimDepend), out, out_to_remote};
  CNodePtr new_out = fprop->NewCNode(new_out_value_input);
  (void)mng->Replace(out_param, new_out);

  // Only in bprop graph, change usage of out to new_out_value.
  // %prefetch = Prefetch(%out)
  // %prefetch_depend = Depend(%prefetch, %dout)
  // %new_out = Depend(%out, %prefetch_depend)
  AnfNodePtrList prefetch_inputs{NewValueNode(prim::kPrimPrefetch), new_out};
  auto prefetch_node = bprop->NewCNodeInFront(prefetch_inputs);
  AnfNodePtrList prefetch_depend_inputs{NewValueNode(prim::kPrimDepend), prefetch_node, dout};
  auto prefetch_depend_node = bprop->NewCNode(prefetch_depend_inputs);
  AnfNodePtrList bprop_out_inputs{NewValueNode(prim::kPrimDepend), new_out, prefetch_depend_node};
  auto bprop_out_node = bprop->NewCNode(bprop_out_inputs);
  const auto &out_value_users = mng->node_users()[new_out];
  AnfNodePtrList make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  for (const auto &user : out_value_users) {
    auto user_node = user.first;
    if (user_node->func_graph() != bprop) {
      continue;
    }
    mng->SetEdge(user_node, user.second, bprop_out_node);
    (void)make_tuple_inputs.emplace_back(user_node);
  }

  // Add Detach after activation is used
  auto make_tuple_node = bprop->NewCNode(make_tuple_inputs);
  auto detach_node = bprop->NewCNode({NewValueNode(prim::kPrimDetach), bprop_out_node, make_tuple_node});
  auto new_output = bprop->NewCNode({NewValueNode(prim::kPrimDepend), bprop->output(), detach_node});
  bprop->set_output(new_output);
  return new_out;
}

void InsertPrefetchForLoad(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(mng);
  MS_EXCEPTION_IF_NULL(func_graph);
  auto tr = mng->Transact();
  constexpr size_t load_ref_index = 1;
  constexpr size_t load_monad_index = 2;
  const AnfNodePtrList &all_nodes = mindspore::TopoSort(func_graph->get_return(), SuccDeeperSimple);
  for (auto node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimLoad)) {
      continue;
    }
    auto cnode = node->cast_ptr<CNode>();
    auto cur_fg = cnode->func_graph();
    MS_EXCEPTION_IF_NULL(cur_fg);
    auto ref_input = cnode->input(load_ref_index);
    auto monad_input = cnode->input(load_monad_index);
    AnfNodePtrList prefetch_inputs{NewValueNode(prim::kPrimPrefetch), ref_input, monad_input};
    auto prefetch_node = cur_fg->NewCNode(prefetch_inputs);
    AnfNodePtrList update_input{NewValueNode(prim::kPrimUpdateState), NewValueNode(kUMonad), prefetch_node};
    auto update_node = cur_fg->NewCNode(update_input);
    (void)tr.SetEdge(node, load_monad_index, update_node);
  }
  tr.Commit();
}
}  // namespace remote_memory
}  // namespace mindspore
