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
#include <regex>
#include "utils/trace_base.h"
#include "ir/core_ops_primitive.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "pipeline/jit/ps/parse/parse.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace remote_memory {
namespace {
struct NodeUserCompare {
  bool operator()(const std::pair<size_t, AnfNodePtr> &lhs, const std::pair<size_t, AnfNodePtr> &rhs) const {
    return lhs.first < rhs.first;
  }
};

using TopoUserList = std::set<std::pair<size_t, AnfNodePtr>, NodeUserCompare>;
using DetachInfo = std::pair<AnfNodePtr, TopoUserList>;
using GradLoadInfo = std::pair<CNodePtr, CNodePtr>;

TopoUserList CollectNodeTopoUserList(const FuncGraphManagerPtr &mng, const AnfNodePtrList &topo_orders,
                                     const AnfNodePtr &node) {
  TopoUserList ret;
  for (const auto &user_node : mng->node_users()[node]) {
    auto cur_iter = std::find(topo_orders.begin(), topo_orders.end(), user_node.first);
    size_t cur_index = std::distance(topo_orders.begin(), cur_iter);
    (void)ret.emplace(std::pair<size_t, AnfNodePtr>(cur_index, *cur_iter));
  }
  return ret;
}

std::vector<DetachInfo> CollectDetachInfo(const FuncGraphManagerPtr &mng, const FuncGraphPtr func_graph,
                                          const CNodePtrList &remote_activation_nodes) {
  const AnfNodePtrList &topo_nodes = mindspore::TopoSort(func_graph->get_return(), SuccDeeperSimple);
  std::vector<DetachInfo> detach_info_list;
  for (const auto &node : remote_activation_nodes) {
    const auto &topo_user_list = CollectNodeTopoUserList(mng, topo_nodes, node);
    (void)detach_info_list.emplace_back(DetachInfo{node, topo_user_list});
  }
  return detach_info_list;
}

std::vector<GradLoadInfo> CollectGradLoadInfo(const FuncGraphManagerPtr &mng, const FuncGraphPtr func_graph,
                                              const CNodePtrList &remote_activation_nodes) {
  // todo: Now, the offset is fixed to 2.
  const AnfNodePtrList &topo_nodes = mindspore::TopoSort(func_graph->get_return(), SuccDeeperSimple);
  AnfNodePtrList topo_cnode;
  for (const auto &node : topo_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    (void)topo_cnode.emplace_back(node);
  }
  std::vector<GradLoadInfo> grad_info_list;
  for (const auto &node : remote_activation_nodes) {
    auto iter = std::find(topo_cnode.begin(), topo_cnode.end(), node);
    if (iter == topo_cnode.end() || std::distance(iter, topo_cnode.end()) < 3) {
      (void)grad_info_list.emplace_back(GradLoadInfo{node, nullptr});
      continue;
    }
    auto grad_load_position_node = *(iter + 2);
    (void)grad_info_list.emplace_back(GradLoadInfo{node, grad_load_position_node->cast<CNodePtr>()});
  }
  return grad_info_list;
}

AnfNodePtr GenerateWrappedCallFgNode(const FuncGraphPtr &wrapped_fg, const FuncGraphPtr &fg,
                                     const AnfNodePtrList &prefetch_elements, const AnfNodePtrList &call_inputs) {
  AnfNodePtrList wrapped_inputs{NewValueNode(fg)};
  // No need to prefetch.
  if (prefetch_elements.empty()) {
    wrapped_inputs.insert(wrapped_inputs.end(), call_inputs.begin(), call_inputs.end());
    return wrapped_fg->NewCNodeInOrder(wrapped_inputs);
  }
  AnfNodePtrList prefetch_results_inputs{NewValueNode(prim::kPrimMakeTuple)};
  for (size_t i = 0; i < prefetch_elements.size(); ++i) {
    auto prefetch_param_node = prefetch_elements[i];
    // todo: Need to add depend node for prefetch.
    // todo: Need to decide whether the remote ops is run synchronously.
    AnfNodePtrList cur_prefetch_inputs{NewValueNode(prim::kPrimPrefetch), prefetch_param_node, NewValueNode(kNone),
                                       NewValueNode(false)};
    auto prefetch_result = wrapped_fg->NewCNodeInOrder(cur_prefetch_inputs);
    (void)prefetch_results_inputs.emplace_back(prefetch_result);
  }
  auto prefetch_result = wrapped_fg->NewCNodeInOrder(prefetch_results_inputs);
  for (size_t i = 0; i < call_inputs.size(); ++i) {
    auto wrapper_depend_input =
      wrapped_fg->NewCNodeInOrder({NewValueNode(prim::kPrimDepend), call_inputs[i], prefetch_result});
    (void)wrapped_inputs.emplace_back(wrapper_depend_input);
  }
  return wrapped_fg->NewCNodeInOrder(wrapped_inputs);
}

AnfNodePtr GenerateWrapperFgReturnNode(const AnfNodePtr &node, const FuncGraphPtr &fg,
                                       const AnfNodePtrList &detach_nodes, bool update) {
  // No elements to detach.
  if (detach_nodes.empty()) {
    return node;
  }

  AnfNodePtr node_after_update = node;
  if (update) {
    // todo: No need to update all parameters? Only requires grad parameters or non-optimizer parameters need.
    AnfNodePtrList update_result_inputs{NewValueNode(prim::kPrimMakeTuple)};
    for (auto detach_node : detach_nodes) {
      auto depend_node = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), node});
      auto update_result =
        fg->NewCNodeInOrder({NewValueNode(prim::kPrimToRemote), detach_node, depend_node, NewValueNode(false)});
      (void)update_result_inputs.emplace_back(update_result);
    }
    auto update_result_node = fg->NewCNodeInOrder(update_result_inputs);
    node_after_update = fg->NewCNodeInOrder({NewValueNode(prim::kPrimDepend), node, update_result_node});
  }
  AnfNodePtrList detach_result_inputs{NewValueNode(prim::kPrimMakeTuple)};
  auto depend_node = fg->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), node_after_update});
  for (auto detach_node : detach_nodes) {
    auto detach_result =
      fg->NewCNodeInOrder({NewValueNode(prim::kPrimDetach), detach_node, depend_node, NewValueNode(MakeValue(false))});
    (void)detach_result_inputs.emplace_back(detach_result);
  }
  auto detach_result_node = fg->NewCNodeInOrder(detach_result_inputs);
  auto wrapper_return_node =
    fg->NewCNodeInOrder({NewValueNode(prim::kPrimDepend), node_after_update, detach_result_node});
  return wrapper_return_node;
}

FuncGraphPtr WrapGraphWithRemoteOps(const FuncGraphPtr &fg, const py::object &prefetch_elements,
                                    const py::object &detach_elements, bool update_detach_elements) {
  MS_EXCEPTION_IF_NULL(fg);
  FuncGraphPtr wrapped_fg = std::make_shared<FuncGraph>();
  if (!py::isinstance<py::tuple>(prefetch_elements) && !py::isinstance<py::list>(prefetch_elements)) {
    MS_LOG(INTERNAL_EXCEPTION) << "prefetch_elements should only be none, tuple or list but got: "
                               << py::str(prefetch_elements);
  }
  auto prefetch_tuple = py::tuple(prefetch_elements);
  parse::Resolver resolver(parse::Parser::GetTopFuncGraph());
  AnfNodePtrList prefetch_list;
  for (size_t i = 0; i < py::len(prefetch_tuple); ++i) {
    auto prefetch_param = resolver.ResolveParameterObj(wrapped_fg, prefetch_tuple[i]);
    (void)prefetch_list.emplace_back(prefetch_param);
  }
  AnfNodePtrList func_graph_call_inputs;
  for (size_t i = 0; i < fg->get_inputs().size(); ++i) {
    (void)func_graph_call_inputs.emplace_back(wrapped_fg->add_parameter());
  }

  if (!py::isinstance<py::tuple>(detach_elements) && !py::isinstance<py::list>(detach_elements)) {
    MS_LOG(INTERNAL_EXCEPTION) << "detach_elements should only be none, tuple or list but got: "
                               << py::str(detach_elements);
  }
  auto detach_tuple = py::tuple(detach_elements);
  AnfNodePtrList detach_list;
  for (size_t i = 0; i < py::len(detach_tuple); ++i) {
    auto detach_param = resolver.ResolveParameterObj(wrapped_fg, detach_tuple[i]);
    (void)detach_list.emplace_back(detach_param);
  }

  auto wrapped_call_node = GenerateWrappedCallFgNode(wrapped_fg, fg, prefetch_list, func_graph_call_inputs);
  auto wrapped_return_node =
    GenerateWrapperFgReturnNode(wrapped_call_node, wrapped_fg, detach_list, update_detach_elements);

  wrapped_fg->set_output(wrapped_return_node);
  return wrapped_fg;
}

bool GetJitEnableRemoteMemoryFromComment(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (node->debug_info() == nullptr) {
    return false;
  }
  const auto &debug_info = trace::GetSourceCodeDebugInfo(node->debug_info());
  const auto &location = debug_info->location();
  if (location == nullptr) {
    MS_LOG(DEBUG) << "Location info is null, node: " << node->DebugString();
    return false;
  }
  const auto &comments = location->comments();
  if (comments.empty()) {
    return false;
  }
  // Only use the last comment.
  const auto &comment = comments.back();
  std::regex regex("^#\\s*@jit.enable_remote_memory\\s*");
  if (std::regex_match(comment, regex)) {
    return true;
  }
  return false;
}

CNodePtrList CollectRemoteActivactionNodes(const FuncGraphPtr &func_graph) {
  const AnfNodePtrList &all_nodes = mindspore::TopoSort(func_graph->get_return(), SuccDeeperSimple);
  CNodePtrList remote_activaction_nodes;
  for (auto node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    // todo: need a unified way to change a node need remote action including functional operations.
    // todo: need to get prefetch offset from comment, now prefetch offset is set to 2.
    if (!GetJitEnableRemoteMemoryFromComment(node)) {
      continue;
    }
    (void)remote_activaction_nodes.emplace_back(node->cast<CNodePtr>());
  }
  return remote_activaction_nodes;
}

void ReplaceCNodePrimitiveWithRemoteMemoryAttr(const FuncGraphManagerPtr &mng, const CNodePtrList &nodes) {
  // todo: Replaced primitive should be stored and reused.
  auto tr = mng->Transact();
  constexpr size_t prim_index = 0;
  for (auto node : nodes) {
    auto prim = GetCNodePrimitive(node);
    auto new_prim = prim->Clone();
    new_prim->set_attr(kRemoteActivationAttr, MakeValue(true));
    tr.SetEdge(node, prim_index, NewValueNode(new_prim));
  }
  tr.Commit();
}

void AddDetachAfterLastForwardUsers(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph,
                                    const CNodePtrList &remote_activation_nodes) {
  const auto &detach_info_list = CollectDetachInfo(mng, func_graph, remote_activation_nodes);
  auto tr = mng->Transact();
  for (const auto &detach_info : detach_info_list) {
    auto cur_node = detach_info.first;
    const auto &topo_user_list = detach_info.second;
    auto last_user_node = (*topo_user_list.rbegin()).second;
    auto last_user_cnode = last_user_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(last_user_cnode);
    auto cur_fg = last_user_cnode->func_graph();
    AnfNodePtrList depend_nodes_input{NewValueNode(prim::kPrimMakeTuple)};
    for (const auto &user : topo_user_list) {
      auto user_node = user.second;
      auto user_cnode = user_node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(user_cnode);
      if (user_cnode->func_graph() != cur_fg) {
        MS_LOG(ERROR) << "Unexpected graph not match for user_cnode: " << user_cnode->DebugString();
        continue;
      }
      (void)depend_nodes_input.emplace_back(user_cnode);
    }
    auto depend_node = cur_fg->NewCNode(depend_nodes_input);
    auto detach_node = cur_fg->NewCNode({NewValueNode(prim::kPrimDetach), cur_node, depend_node, NewValueNode(false)});
    auto new_last_user_node = cur_fg->NewCNode({NewValueNode(prim::kPrimDepend), last_user_node, detach_node});
    tr.Replace(last_user_node, new_last_user_node);
  }
  tr.Commit();
}

void AddGradLoad(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph, const CNodePtrList &nodes) {
  const auto &grad_load_info_list = CollectGradLoadInfo(mng, func_graph, nodes);
  auto tr = mng->Transact();
  for (const auto &grad_load_info : grad_load_info_list) {
    auto fetch_node = grad_load_info.first;
    auto fetch_position_node = grad_load_info.second;
    if (fetch_position_node == nullptr) {
      MS_LOG(ERROR) << "Can not find fetch position node for node: " << fetch_node->DebugString();
      continue;
    }
    if (fetch_node->func_graph() != fetch_position_node->func_graph()) {
      // todo: Need to ensure fetch node and fetch position node in the same graph later.
      MS_LOG(ERROR) << "Should in the same graph for fetch node: " << fetch_node->DebugString()
                    << " and fetch position node: " << fetch_position_node->DebugString();
      continue;
    }
    auto fg = fetch_position_node->func_graph();
    AnfNodePtrList grad_load_node_inputs{NewValueNode(prim::kPrimGradLoad), fetch_position_node, fetch_node,
                                         NewValueNode(kNone), NewValueNode(false)};
    auto grad_load_node = fg->NewCNode(grad_load_node_inputs);
    tr.Replace(fetch_position_node, grad_load_node);
  }
  tr.Commit();
}

}  // namespace

bool IsEnableGradOffload(const py::object &obj) {
  if (!py::hasattr(obj, kEnableGradOffloadAttr)) {
    return false;
  }
  return py::getattr(obj, kEnableGradOffloadAttr).cast<bool>();
}

void SetEnableGradOffloadToAbstract(const AbstractBasePtr &abs) {
  abs->set_user_data<bool>(kEnableGradOffloadAttr, std::make_shared<bool>(true));
}

bool IsEnableGradOffloadAbstract(const AbstractBasePtr &abs) {
  if (!abs->has_user_data(kEnableGradOffloadAttr)) {
    return false;
  }
  return *(abs->user_data<bool>(kEnableGradOffloadAttr));
}

CNodePtr ActivationToRemote(const FuncGraphPtr &fprop, const AnfNodePtr &activaction) {
  AnfNodePtrList inputs{NewValueNode(prim::kPrimToRemote), activaction, NewValueNode(kNone), NewValueNode(false)};
  return fprop->NewCNode(inputs);
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
    AnfNodePtrList prefetch_inputs{NewValueNode(prim::kPrimPrefetch), ref_input, monad_input, NewValueNode(false)};
    auto prefetch_node = cur_fg->NewCNode(prefetch_inputs);
    AnfNodePtrList update_input{NewValueNode(prim::kPrimUpdateState), NewValueNode(kUMonad), prefetch_node};
    auto update_node = cur_fg->NewCNode(update_input);
    (void)tr.SetEdge(node, load_monad_index, update_node);
  }
  tr.Commit();
}

FuncGraphPtr GenerateMultitypeFGWithRemoteOps(const FuncGraphPtr &func_graph, const TypePtrList &types) {
  MS_EXCEPTION_IF_NULL(func_graph);
  size_t input_size = func_graph->get_inputs().size();
  if (input_size + 1 != types.size()) {
    MS_LOG(INTERNAL_EXCEPTION) << "Size not matched, graph input size is: " << std::to_string(input_size)
                               << ", types size is: " << types.size();
  }
  auto wrapped_fg = std::make_shared<FuncGraph>();
  AnfNodePtrList detach_list;
  for (size_t i = 0; i < input_size; ++i) {
    auto cur_input = wrapped_fg->add_parameter();
    MS_EXCEPTION_IF_NULL(types[i]);
    if (types[i]->isa<TensorType>()) {
      (void)detach_list.emplace_back(cur_input);
    }
  }
  auto prefetch_type = types.back();
  MS_EXCEPTION_IF_NULL(prefetch_type);
  auto prefetch_tuple_type = prefetch_type->cast<TuplePtr>();
  MS_EXCEPTION_IF_NULL(prefetch_tuple_type);
  auto prefetch_size = prefetch_tuple_type->elements().size();
  auto prefetch_input = wrapped_fg->add_parameter();
  AnfNodePtrList prefetch_list;
  for (size_t i = 0; i < prefetch_size; ++i) {
    auto cur_detach_element =
      wrapped_fg->NewCNode({NewValueNode(prim::kPrimTupleGetItem), prefetch_input, NewValueNode(int64_t(i))});
    (void)prefetch_list.emplace_back(cur_detach_element);
  }
  MS_EXCEPTION_IF_CHECK_FAIL((func_graph->get_inputs().size() + 1) == wrapped_fg->get_inputs().size(),
                             "size not matched");
  AnfNodePtrList func_graph_call_inputs;
  for (size_t i = 0; i < wrapped_fg->get_inputs().size() - 1; ++i) {
    (void)func_graph_call_inputs.emplace_back(wrapped_fg->get_inputs()[i]);
  }
  auto call_node = GenerateWrappedCallFgNode(wrapped_fg, func_graph, prefetch_list, func_graph_call_inputs);
  auto return_node = GenerateWrapperFgReturnNode(call_node, wrapped_fg, detach_list, true);
  wrapped_fg->set_output(return_node);
  return wrapped_fg;
}

void AddRemoteOpsToGraphs(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph) {
  auto tr = mng->Transact();
  const AnfNodePtrList &all_nodes = mindspore::TopoSort(func_graph->get_return(), SuccDeeperSimple);
  for (auto node : all_nodes) {
    if (!node->isa<CNode>()) {
      continue;
    }
    constexpr size_t prim_index = 0;
    constexpr auto remote_memory_info = "remote_memory_info";
    auto cnode = node->cast_ptr<CNode>();
    auto prim_fg = GetValueNode<FuncGraphPtr>(cnode->input(prim_index));
    if (prim_fg == nullptr) {
      continue;
    }
    if (!prim_fg->has_attr(remote_memory_info)) {
      continue;
    }
    MS_LOG(INFO) << "Wrap remote ops for function graph: " << prim_fg->ToString();
    auto remote_memory_info_obj = prim_fg->get_attr(remote_memory_info)->cast<parse::PyObjectWrapperPtr>()->obj();
    auto prefetch_list = py::getattr(remote_memory_info_obj, "prefetch");
    auto detach_list = py::getattr(remote_memory_info_obj, "detach");
    auto new_prim_fg = WrapGraphWithRemoteOps(prim_fg, prefetch_list, detach_list, false);
    tr.SetEdge(node, prim_index, NewValueNode(new_prim_fg));
  }
  tr.Commit();
}

bool InsertActivactionRemoteOpsForGraph(const FuncGraphManagerPtr &mng, const FuncGraphPtr &func_graph) {
  const AnfNodePtrList &all_nodes = mindspore::TopoSort(func_graph->get_return(), SuccDeeperSimple);
  const auto &remote_activation_nodes = CollectRemoteActivactionNodes(func_graph);
  if (remote_activation_nodes.empty()) {
    return false;
  }
  ReplaceCNodePrimitiveWithRemoteMemoryAttr(mng, remote_activation_nodes);
  AddDetachAfterLastForwardUsers(mng, func_graph, remote_activation_nodes);
  AddGradLoad(mng, func_graph, remote_activation_nodes);
  return true;
}
}  // namespace remote_memory
}  // namespace mindspore
