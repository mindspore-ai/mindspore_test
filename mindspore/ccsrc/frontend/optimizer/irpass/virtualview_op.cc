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

#include <map>
#include <string>
#include <vector>
#include <utility>
#include "frontend/optimizer/irpass/view_inplace_utils.h"

namespace mindspore {
namespace opt {
namespace irpass {

void VirtualViewInsertProcesser::Run() {
  for (const auto &node : TopoSort(func_graph_->get_return())) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }

    // Insert VirtualView op if input is viewed and changed
    CheckAndInsertVirtualViewOp(cnode);

    if (IsViewNode(cnode)) {
      ProcessViewNode(cnode);
      continue;
    }

    if (IsInplaceNode(cnode)) {
      ProcessInplaceNode(cnode);
    }
  }

  DoVirtualViewInputReplace();
}

AnfNodePtr VirtualViewInsertProcesser::CreateVirtualViewNode(const CNodePtr &view_output, AnfNodePtr *last_umonad) {
  const auto &inputs = view_output->inputs();
  AnfNodePtrList new_inputs(inputs.begin(), inputs.end() - 1);
  new_inputs.push_back(*last_umonad);

  auto virtual_view_node = func_graph_->NewCNodeInOrder(new_inputs);
  virtual_view_node->set_abstract(view_output->abstract());
  virtual_view_node->AddAttr(kIsVirtualViewOp, MakeValue(true));

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

void VirtualViewInsertProcesser::VirtualViewInsertAction(const CNodePtr &cnode, const CNodePtr &view_cnode) {
  MS_EXCEPTION_IF_NULL(cnode);
  MS_EXCEPTION_IF_NULL(view_cnode);

  AnfNodePtr umonad = cnode->inputs().back();
  const auto &umonad_abstract = umonad->abstract();
  if (umonad_abstract == nullptr || !umonad_abstract->isa<abstract::AbstractUMonad>()) {
    MS_LOG(EXCEPTION) << "Invalid cnode, should have umonad as the last input, but got cnode: "
                      << umonad->DebugString();
  }

  auto view_chain_it = view_chains_.find(view_cnode);
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

void VirtualViewInsertProcesser::UpdateViewChains(const CNodePtr &view_cnode) {
  if (view_cnode->inputs().size() < 2) {
    MS_LOG(DEBUG) << "Invalid viewed cnode, should have at least one input: " << view_cnode->DebugString();
    return;
  }

  auto input_node = view_cnode->input(1);
  if (view_chains_.count(input_node)) {
    // n = View(m)
    // view_chains[m] = [m]
    // view_chains_: {m: [m], n: [m, n]}
    auto new_chain = view_chains_[input_node];
    new_chain.push_back(view_cnode);
    view_chains_[view_cnode] = std::move(new_chain);
  } else {
    // m = View(y)
    // view_chains_: {}
    // view_chains_: {m: [m]}
    view_chains_[view_cnode] = {view_cnode};
  }
}

AnfNodePtr VirtualViewInsertProcesser::FindViewRootNode(const CNodePtr &view_cnode) {
  // m = View(y), n = View(m)
  // view_chains_: {m: [m], n: [m, n]}
  // [m, n] --> m
  // m --> y
  auto chain_it = view_chains_.find(view_cnode);
  if (chain_it == view_chains_.end()) {
    MS_LOG(DEBUG) << "View chain not found for node: " << view_cnode->DebugString();
    return nullptr;
  }

  const auto &view_chain = chain_it->second;
  if (view_chain.empty()) {
    MS_LOG(DEBUG) << "Empty view chain for node: " << view_cnode->DebugString();
    return nullptr;
  }

  AnfNodePtr front_node = view_chain.front();
  auto root_view_node = front_node->cast<CNodePtr>();
  if (root_view_node == nullptr) {
    MS_LOG(DEBUG) << "Root view node is not a CNode, view_cnode:" << view_cnode->DebugString();
    return nullptr;
  }

  if (root_view_node->inputs().size() < 2) {
    MS_LOG(DEBUG) << "Invalid root_view_node, should have at least one input: " << root_view_node->DebugString();
    return nullptr;
  }

  return root_view_node->input(1);
}

void VirtualViewInsertProcesser::ProcessViewNode(const CNodePtr &cnode) {
  // m = View(y), n = View(m)
  // ---> view_chains: {m: [m], n: [m, n]}
  // ---> view_dependencies: {m: y, n: y}
  // ---> view_modifications: {y: {m: false, n: false}}
  UpdateViewChains(cnode);
  auto root_node = FindViewRootNode(cnode);
  MS_EXCEPTION_IF_NULL(root_node);
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
  for (const auto &input : cnode->inputs()) {
    auto input_node = input->cast<CNodePtr>();
    if (input_node == nullptr) {
      continue;
    }

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
  std::map<AnfNodePtr, AnfNodePtr> virtual_view_input;
  auto manager = func_graph_->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (auto node : TopoSort(func_graph_->return_node())) {
    if (!irpass::IsCNode(node)) {
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
      const auto &abs = node->abstract();
      MS_EXCEPTION_IF_NULL(abs);
      auto ref = abs->cast<abstract::AbstractRefPtr>();
      if (ref == nullptr) {
        MS_LOG(EXCEPTION) << "The virtual view op abstract is not ref: " << abs->ToString()
                          << ", virtual view operation is: " << node->DebugString();
      }
      auto view_op = abs->user_data<CNode>(kOriginalViewOp);
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

bool VirtualViewInsertProcesser::IsVirtualViewCNode(const AnfNodePtr &node) {
  auto cnode = node->cast<CNodePtr>();
  return cnode != nullptr && cnode->HasAttr(kIsVirtualViewOp);
}

void VirtualViewInsertProcesser::DoVirtualViewInputReplace() {
  const auto &all_nodes = TopoSort(func_graph_->return_node(), SuccDeeperSimple);
  bool exist_virtual_view_nodes = std::any_of(all_nodes.begin(), all_nodes.end(),
                                              [this](const AnfNodePtr &node) { return IsVirtualViewCNode(node); });
  if (!exist_virtual_view_nodes) {
    return;
  }

  ChangeVirtualViewInputInner();
}

void VirtualViewInsert(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  MS_EXCEPTION_IF_NULL(root);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);

  VirtualViewInsertProcesser(root, manager).Run();
  auto sub_graphs = root->func_graphs_used_total();
  for (const auto &sub_graph : sub_graphs) {
    VirtualViewInsertProcesser(sub_graph, manager).Run();
  }

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("opt_insert_virtualview.ir", root);
  }
#endif
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
