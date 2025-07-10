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

#include "frontend/optimizer/irpass/virtualview_op.h"

#include "frontend/optimizer/irpass/virtualviewgrad_op.h"
#include "frontend/optimizer/irpass/view_inplace_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {

void VirtualViewInsertProcesser::Run() {
  InitViewInfoFromParams();

  for (const auto &node : TopoSort(func_graph_->get_return())) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr || IsPrimitiveCNode(node, prim::kPrimVirtualViewGrad) ||
        IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      continue;
    }

    // Insert VirtualView op if input is viewed and changed
    CheckAndInsertVirtualViewOp(cnode);

    if (IsViewNode(cnode)) {
      ProcessViewNode(cnode);
    } else if (IsInplaceNode(cnode)) {
      ProcessInplaceNode(cnode);
    }
  }

  DoVirtualViewInputReplace();
}

AnfNodePtr VirtualViewInsertProcesser::ReplaceWithParameter(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!is_viewed_param_existed_) {
    return node;
  }
  auto refkey = GetRefKey(node);
  if (refkey.empty()) {
    return node;
  }

  auto it = refkey_to_param_.find(refkey);
  if (it != refkey_to_param_.end()) {
    return it->second;
  }

  return node;
}

std::pair<AnfNodePtr, AnfNodePtrList> VirtualViewInsertProcesser::GetViewInfo(const AnfNodePtr &param) {
  MS_EXCEPTION_IF_NULL(param);
  AnfNodePtr current_node = param;
  AnfNodePtrList view_chain;
  CNodePtr view_node;
  while (true) {
    // m = View(y), n = View(m), u = View(n)
    // 1) param_x
    // param_x ---> nullptr
    // return {param_x, []}
    // 2) param_m:
    // param_m ---> m ---> [param_m] ---> y
    // y ---> nullptr
    // return {y, [param_m]}
    // 3) param_u
    // param_u ---> u ---> [param_u] ---> n
    // n ---> n ---> [n, param_u] ---> m
    // m ---> m ---> [param, n, param_u]
    // return {y, [param, n, param_u]}
    view_node = std::get<0>(IsCreatedByViewOp(current_node));
    if (view_node == nullptr) {
      MS_LOG(DEBUG) << "GetViewInfo param: " << param->DebugString() << " view_chain.size(): " << view_chain.size();
      return {ReplaceWithParameter(current_node), std::move(view_chain)};
    }

    auto view_node_param = ReplaceWithParameter(view_node);

    auto it_chain = view_chains_.find(view_node_param);
    auto it_dep = view_dependencies_.find(view_node_param);
    if (it_chain != view_chains_.end() && it_dep != view_dependencies_.end()) {
      const auto &existing_chain = it_chain->second;
      view_chain.insert(view_chain.begin(), existing_chain.begin(), existing_chain.end());
      MS_LOG(DEBUG) << "GetViewInfo from existing chain param: " << param->DebugString()
                    << " view_chain.size(): " << view_chain.size()
                    << " view_node_param: " << view_node_param->DebugString();
      return {it_dep->second, std::move(view_chain)};
    } else {
      view_chain.insert(view_chain.begin(), view_node_param);
    }
    MS_EXCEPTION_IF_CHECK_FAIL(view_node->inputs().size() > kIndex1,
                               "Input size should be larger than 1, view_node: " + view_node->DebugString());
    current_node = view_node->input(1);
  }
}

void VirtualViewInsertProcesser::InitViewInfoFromParams() {
  if (!is_viewed_param_existed_) {
    return;
  }

  for (const auto &param : params_) {
    auto refkey = GetRefKey(param);
    if (refkey.empty()) {
      continue;
    }
    refkey_to_param_[refkey] = param;
  }

  bool is_viewed_param_existed = false;
  for (const auto &param : params_) {
    MS_EXCEPTION_IF_NULL(param);
    auto [root_node, view_chain] = GetViewInfo(param);
    if (root_node == nullptr || view_chain.empty()) {
      MS_LOG(DEBUG) << "Fail to get view info from param: " << param->DebugString();
      continue;
    }

    is_viewed_param_existed = true;
    view_chains_[param] = view_chain;
    view_dependencies_[param] = root_node;
    view_modifications_[root_node][param] = false;
  }
  is_viewed_param_existed_ = is_viewed_param_existed;
  MS_LOG(DEBUG) << "is_viewed_param_existed_ updated: " << is_viewed_param_existed_;
}

AnfNodePtr VirtualViewInsertProcesser::CreateVirtualViewNode(const AnfNodePtr &view_output, AnfNodePtr *last_umonad) {
  const auto &view_node = std::get<0>(IsCreatedByViewOp(view_output));
  MS_EXCEPTION_IF_NULL(view_node);
  MS_EXCEPTION_IF_CHECK_FAIL(view_node->inputs().size() > kIndex2,
                             "Input size should be larger than 2, view_node: " + view_node->DebugString());
  const auto &inputs = view_node->inputs();
  AnfNodePtrList new_inputs(inputs.begin(), inputs.end() - 1);
  auto &input = new_inputs[1];
  input = ReplaceWithParameter(input);
  new_inputs.push_back(*last_umonad);

  auto virtual_view_node = func_graph_->NewCNodeInOrder(new_inputs);
  virtual_view_node->set_abstract(view_node->abstract());
  virtual_view_node->AddAttr(kIsVirtualViewOp, MakeValue(true));
  virtual_view_node->set_user_data<AnfNode>(kIsVirtualViewOp, view_output);

  auto new_umonad =
    func_graph_->NewCNodeInOrder({NewValueNode(prim::kPrimUpdateState), *last_umonad, virtual_view_node});
  new_umonad->set_abstract((*last_umonad)->abstract());
  *last_umonad = new_umonad;
  return virtual_view_node;
}

void VirtualViewInsertProcesser::ResetViewModificationStatus(const AnfNodePtr &view_output) {
  // view_dependencies: {m: y, n: y, ...}
  // view_modifications: {y: {m: true, n: true, ...}, ...}
  // Reset view_modifications: {y: {m: false, n: true, ...}, ...} after VirtualView of m inserted
  auto dep_it = view_dependencies_.find(view_output);
  if (dep_it == view_dependencies_.end()) {
    return;
  }

  auto mod_it = view_modifications_.find(dep_it->second);
  if (mod_it == view_modifications_.end()) {
    return;
  }

  auto &view_status_map = mod_it->second;
  auto status_it = view_status_map.find(view_output);
  if (status_it != view_status_map.end()) {
    status_it->second = false;
    MS_LOG(DEBUG) << "Reset view modification status for: " << view_output->DebugString();
  }
}

void VirtualViewInsertProcesser::VirtualViewInsertAction(const CNodePtr &cnode, const AnfNodePtr &view_node) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(view_node);

  AnfNodePtr umonad = cnode->inputs().back();
  (void)CheckUMonad(umonad);

  auto view_chain_it = view_chains_.find(view_node);
  if (view_chain_it == view_chains_.end()) {
    return;
  }

  AnfNodePtr first_virtual_view_node = nullptr;
  AnfNodePtr first_new_umonad = nullptr;
  AnfNodePtr last_umonad = umonad;

  for (auto view_output : view_chain_it->second) {
    auto virtual_view_node = CreateVirtualViewNode(view_output, &last_umonad);
    if (first_virtual_view_node == nullptr) {
      first_virtual_view_node = virtual_view_node;
      first_new_umonad = last_umonad;
    }

    ResetViewModificationStatus(view_output);
  }

  // SetEdge for original umonad users to last_umonad
  auto updatastate_users = manager_->node_users()[umonad];
  for (const auto &node_index : updatastate_users) {
    auto used_node = node_index.first;
    MS_EXCEPTION_IF_NULL(used_node);
    if (used_node == first_virtual_view_node || used_node == first_new_umonad) {
      continue;
    }
    auto used_cnode = used_node->cast<CNodePtr>();
    manager_->SetEdge(used_cnode, node_index.second, last_umonad);
  }
}

void VirtualViewInsertProcesser::UpdateViewModificationStatus(const AnfNodePtr &input_node) {
  // m = View(y), n = View(y)
  // view_dependencies: {m: y, n: y, ...}
  // view_modifications: {y: {m: false, n: false, ...}, ...}
  // Update view_modifications: {y: {m: true, n: true, ...}, ...} if y or one of y viewed node changed
  auto mod_it = view_modifications_.find(input_node);
  if (mod_it == view_modifications_.end()) {
    // input_node is y viewed node like m, n
    auto dep_it = view_dependencies_.find(input_node);
    if (dep_it == view_dependencies_.end()) {
      return;
    }
    mod_it = view_modifications_.find(dep_it->second);
    if (mod_it == view_modifications_.end()) {
      return;
    }
  }
  auto &view_status_map = mod_it->second;
  for (auto &view_status : view_status_map) {
    view_status.second = true;
  }
}

void VirtualViewInsertProcesser::ProcessViewNode(const CNodePtr &cnode) {
  // m = View(y), n = View(m)
  // ---> view_chains: {m: [m], n: [m, n]}
  // ---> view_dependencies: {m: y, n: y}
  // ---> view_modifications: {y: {m: false, n: false}}
  auto [root_node, view_chain] = GetViewInfo(cnode);
  MS_EXCEPTION_IF_NULL(root_node);
  MS_EXCEPTION_IF_CHECK_FAIL(!view_chain.empty(),
                             "View node's chain should contain itself, cnode: " + cnode->DebugString());
  view_chains_[cnode] = view_chain;
  view_dependencies_[cnode] = root_node;
  view_modifications_[root_node][cnode] = false;
}

void VirtualViewInsertProcesser::ProcessInplaceNode(const CNodePtr &cnode) {
  auto prim = GetCNodePrimitive(cnode);
  const auto &inplace_indexes = prim->rw_write_input_indexes();
  for (size_t index = 0; index < inplace_indexes.size(); ++index) {
    auto input_node = cnode->input(inplace_indexes[index] + 1);
    UpdateViewModificationStatus(input_node);
  }
}

void VirtualViewInsertProcesser::CheckAndInsertVirtualViewOp(const CNodePtr &cnode) {
  // When viewed output m is used in operation
  // view_dependencies: {m: y} --> y
  // view_modifications: {y: {m: true}} --> true
  // Insert VirtualView op of m
  for (const auto &input_node : cnode->inputs()) {
    auto dep_it = view_dependencies_.find(input_node);
    if (dep_it == view_dependencies_.end()) {
      continue;
    }
    const auto &view_input = dep_it->second;
    auto mod_it = view_modifications_.find(view_input);
    if (mod_it == view_modifications_.end()) {
      continue;
    }
    const auto &view_status_map = mod_it->second;
    auto status_it = view_status_map.find(input_node);
    if (status_it != view_status_map.end() && status_it->second) {
      VirtualViewInsertAction(cnode, input_node);
    }
  }
}

void VirtualViewInsertProcesser::ChangeVirtualViewInputInner() {
  std::unordered_map<AnfNodePtr, AnfNodePtr> virtual_view_input;
  auto manager = func_graph_->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto node : TopoSort(func_graph_->return_node())) {
    if (!irpass::IsCNode(node) || node->func_graph() != func_graph_) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();

    for (size_t i = 1; i < cnode->size(); i++) {
      auto original_input = cnode->input(i);
      if (virtual_view_input.count(original_input) == 0) {
        continue;
      }
      // Find the final virtual view cnode to replace
      // For example:
      // %1 = View(%0)
      // %2 = VirtualView(%1)
      // ...
      // %3 = VirtualView(%1)
      // %4 Depend(%1, U)==> %4 = Depend(%3, U)
      AnfNodePtr repalced_node = virtual_view_input[original_input];
      while (virtual_view_input.count(repalced_node) != 0) {
        repalced_node = virtual_view_input[repalced_node];
      }
      MS_LOG(INFO) << "Replace cnode : " << cnode->DebugString() << " input from: " << original_input->DebugString()
                   << " to: " << repalced_node->DebugString() << " for VirtualView ops replacement.";
      manager->SetEdge(cnode, i, repalced_node);
    }

    if (cnode->HasAttr(kIsVirtualViewOp)) {
      auto view_op = cnode->user_data<AnfNode>(kIsVirtualViewOp);
      if (view_op == nullptr) {
        MS_LOG(INFO) << "The virtual view op has no user data: " << node->DebugString();
        continue;
      }
      // This will either insert or find the existing entry
      // %1 = View(%0)
      // %2 = VirtualView(%1)
      // {%1: %2}
      auto &entry = virtual_view_input[view_op];
      if (entry == nullptr) {
        entry = cnode;
        MS_LOG(INFO) << "Record cnode as virtual view node: " << cnode->DebugString();
        continue;
      }

      // Follow the chain to find the last node
      // %1 = View(%0)
      // %2 = VirtualView(%1)
      // ...
      // %3 = VirtualView(%1)
      // {%1: %2, %2: %3}
      auto *replaced_node = &entry;
      while (virtual_view_input.count(*replaced_node)) {
        replaced_node = &virtual_view_input[*replaced_node];
      }
      virtual_view_input[*replaced_node] = cnode;
      MS_LOG(INFO) << "Record cnode as virtual view node: " << cnode->DebugString();
    }
  }
}

void VirtualViewInsertProcesser::DoVirtualViewInputReplace() {
  const auto &all_nodes = TopoSort(func_graph_->return_node());
  bool exist_virtual_view_nodes = std::any_of(all_nodes.begin(), all_nodes.end(),
                                              [this](const AnfNodePtr &node) { return IsVirtualViewCNode(node); });
  if (!exist_virtual_view_nodes) {
    return;
  }

  ChangeVirtualViewInputInner();
}

bool VirtualViewInsert(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  MS_EXCEPTION_IF_NULL(root);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);

  PreprocessForVirtualViewGradInsert(root, opt);

  VirtualViewInsertProcesser(root, manager, false).Run();
  auto sub_graphs = root->func_graphs_used_total();
  for (const auto &sub_graph : sub_graphs) {
    VirtualViewInsertProcesser(sub_graph, manager, true).Run();
  }
  return false;
}

AnfNodePtr VirtualViewEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  auto func_graph = node->func_graph();
  if (!IsVirtualViewCNode(node) || func_graph == nullptr) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  auto original_cnode = cnode->user_data<AnfNode>(kIsVirtualViewOp);
  MS_EXCEPTION_IF_NULL(original_cnode);
  auto depend_node =
    func_graph->NewCNode({NewValueNode(prim::kPrimDepend), original_cnode, CheckUMonad(cnode->inputs().back())});
  depend_node->set_abstract(original_cnode->abstract());
  return depend_node;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
