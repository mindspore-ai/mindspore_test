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

#include "frontend/optimizer/irpass/virtualviewgrad_op.h"

#include <map>
#include <string>
#include <vector>
#include <utility>
#include "frontend/optimizer/irpass/view_inplace_utils.h"
#include "mindspore/ops/op_def/other_ops.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
constexpr auto kOriginalViewOp = "view_op";
void InsertVirtualViewGradInner(const FuncGraphPtr &func_graph, const CNodePtr &view_cnode, const AnfNodePtr &umonad,
                                const FuncGraphManagerPtr &manager) {
  // Insert VirtualViewGrad op recursively
  // eg:
  // CNode1 = PrimFunc_InplaceAddExt(x_view_output2, 1, U1)
  // U2 = UpdateState(U1, CNode1)
  // ==>
  // ...
  // CNode2 = VirtualViewGrad(x_view_input2, x_view_output2, U2)
  // U3 = UpdateState(U2, CNode2)
  // CNode3 = VirtualViewGrad(x_view_input1, x_view_input2(x_view_output1), U3)
  // U4 = UpdateState(U3, CNode3)
  CNodePtr view_output = view_cnode;
  AnfNodePtr last_umonad = umonad;
  AnfNodePtr first_virtual_view_grad_node = nullptr;
  AnfNodePtr first_new_umonad = nullptr;
  while (true) {
    const auto &view_output_node_inputs = view_output->inputs();
    auto view_input = view_output_node_inputs[1];
    const auto &ori_view_op = GetCNodePrimitive(view_output)->Clone();
    auto view_op_node = NewValueNode(ori_view_op);
    // To calculate dout for view_input and view_output, insert origin view cnode inputs:
    // ==> view_output = {kPrimViewOp, view_input, other_view_arg1, other_view_arg2, ..., U_for_view}
    // ==> From: VirtualViewGrad(view_input, view_output, U_for_virtual_view_grad)
    // ==> To: VirtualViewGrad(view_input, view_output, kPrimViewOp, other_view_arg1, other_view_arg2, ...,
    // U_for_virtual_view_grad)
    AnfNodePtrList vvg_node_inputs = {NewValueNode(prim::kPrimVirtualViewGrad), view_input, view_output, view_op_node};
    for (size_t i = kIndex2; i < view_output_node_inputs.size() - 1; ++i) {
      (void)vvg_node_inputs.emplace_back(view_output_node_inputs[i]);
    }
    (void)vvg_node_inputs.emplace_back(last_umonad);
    auto vvg_node = func_graph->NewCNodeInOrder(vvg_node_inputs);
    vvg_node->set_abstract(view_input->abstract());
    auto new_umonad = func_graph->NewCNodeInOrder({NewValueNode(prim::kPrimUpdateState), last_umonad, vvg_node});
    new_umonad->set_abstract(last_umonad->abstract());
    if (first_virtual_view_grad_node == nullptr) {
      first_virtual_view_grad_node = vvg_node;
      first_new_umonad = new_umonad;
    }
    last_umonad = new_umonad;
    auto result = IsCreatedByViewOp(view_input);
    if (result.first == nullptr) {
      break;
    }
    view_output = result.first;
  }
  // Set edge for original umonad users to last_umonad
  auto updatastate_users = manager->node_users()[umonad];
  for (const auto &node_index : updatastate_users) {
    auto used_node = node_index.first;
    MS_EXCEPTION_IF_NULL(used_node);
    if (used_node == first_virtual_view_grad_node || used_node == first_new_umonad) {
      continue;
    }
    auto used_cnode = used_node->cast<CNodePtr>();
    manager->SetEdge(used_cnode, node_index.second, last_umonad);
  }
}

void InsertVirtualViewGradAfterInplaceCNode(const CNodePtr &inplace_cnode, const CNodePtr &view_output_cnode,
                                            const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(inplace_cnode);
  MS_EXCEPTION_IF_NULL(view_output_cnode);
  MS_EXCEPTION_IF_NULL(func_graph);
  // CNode1 = PrimFunc_Inplace(x, y, inplace_umonad)
  // inplace_next_updatestate = UpdateState(inplace_umonad, CNode1)
  AnfNodePtr inplace_next_updatestate = nullptr;
  AnfNodePtr inplace_umonad = inplace_cnode->inputs().back();
  const auto &inplace_umonad_abstract = inplace_umonad->abstract();
  if (inplace_umonad_abstract == nullptr || !inplace_umonad_abstract->isa<abstract::AbstractUMonad>()) {
    MS_LOG(EXCEPTION) << "Invalid inplace cnode, should have umonad as the last input, but got cnode: "
                      << inplace_cnode->DebugString();
  }
  for (const auto &node_index : manager->node_users()[inplace_cnode]) {
    const auto &used_node = node_index.first;
    MS_EXCEPTION_IF_NULL(used_node);
    if (!IsPrimitiveCNode(used_node, prim::kPrimUpdateState)) {
      continue;
    }
    auto update_cnode = used_node->cast<CNodePtr>();
    if (update_cnode->input(1) == inplace_umonad && update_cnode->input(2) == inplace_cnode) {
      inplace_next_updatestate = used_node;
      break;
    }
  }
  MS_EXCEPTION_IF_NULL(inplace_next_updatestate);
  InsertVirtualViewGradInner(func_graph, view_output_cnode, inplace_next_updatestate, manager);
}

void VirtualViewGradInsertInner(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  for (const auto &node : TopoSort(func_graph->get_return())) {
    auto cnode = node->cast<CNodePtr>();
    if (cnode == nullptr) {
      continue;
    }
    auto prim = GetCNodePrimitive(cnode);
    if (prim == nullptr || !prim->inplace_prim()) {
      continue;
    }
    CNodePtr view_node;
    bool is_view_output;
    const auto &inplace_indexes = prim->rw_write_input_indexes();
    for (size_t index = 0; index < inplace_indexes.size(); ++index) {
      auto input_node = cnode->input(inplace_indexes[index] + 1);
      std::tie(view_node, is_view_output) = IsCreatedByViewOp(input_node);
      // ViewGradTodo: find real view_node
      // 1. If view_node not nullptr, do insert VirtualViewGrad
      // 2. If view_node is nullptr, but is_view_output is true, throw exception, not support control flow
      // 3. If view_node is nullptr, and is_view_output is false, inplace input is not a view output, just ignore
      if (view_node == nullptr) {
        if (is_view_output) {
          MS_LOG(WARNING) << "Inplace modification of the output of view op is not supported in control flow.";
        }
        continue;
      }
      (void)InsertVirtualViewGradAfterInplaceCNode(cnode, view_node, func_graph, manager);
    }
  }
}

void RemoveRedundantVirtualViewGradInner(const FuncGraphPtr &func_graph, const FuncGraphManagerPtr &manager) {
  // Remove virtualviewgrad op only used by updatestate
  auto &node_users = manager->node_users();
  constexpr size_t kMinUsersSize = 1;
  for (auto &node : TopoSort(func_graph->get_return())) {
    if (!IsPrimitiveCNode(node, prim::kPrimVirtualViewGrad)) {
      continue;
    }
    auto &cur_node_users = node_users[node];
    if (cur_node_users.size() == kMinUsersSize &&
        IsPrimitiveCNode(cur_node_users.front().first, prim::kPrimUpdateState)) {
      // Remove redundant op, replace %2 to U
      // %1 = VirtualViewGrad(%0, x, ..., U)
      // %2 = UpdateState(U, %1)
      const auto &use_node = cur_node_users.front().first;
      manager->Replace(use_node, use_node->cast<CNodePtr>()->input(kIndex1));
    }
  }
}

bool CheckControlFlow(const PrimitivePtr &prim, const CNodePtr &cnode) {
  for (auto index : prim->rw_write_input_indexes()) {
    const auto &input = cnode->input(prim->rw_write_input_indexes()[index] + 1);
    const auto &input_abs = input->abstract();
    if (!input_abs->isa<abstract::AbstractRefTensor>()) {
      MS_LOG(EXCEPTION) << "The rw_write input of inplace op abstract is not ref:" << input_abs->ToString()
                        << ", inplace operation is: " << cnode->DebugString();
    }
    auto input_ref = input_abs->cast<abstract::AbstractRefPtr>();
    if (input_ref->is_view_output()) {
      auto view_op = input_ref->user_data<CNode>(kOriginalViewOp);
      if (view_op == nullptr || view_op->func_graph() != cnode->func_graph()) {
        return true;
      }
    }
  }
  return false;
}

void MarkViewOp(const AnfNodePtr &node, bool *control_flow_scene) {
  if (IsViewNode(node)) {
    const auto &abs = node->abstract();
    MS_EXCEPTION_IF_NULL(abs);
    auto ref = abs->cast<abstract::AbstractRefPtr>();
    if (ref == nullptr) {
      MS_LOG(EXCEPTION) << "The view op abstract is not ref:" << ref->ToString()
                        << ", view operation is: " << node->DebugString();
    }
    auto cnode = node->cast<CNodePtr>();
    ref->set_user_data<CNode>(kOriginalViewOp, cnode);
    return;
  }
  if (!IsInplaceNode(node)) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
  // Check if is control flow scene with view inplace.
  if (CheckControlFlow(prim, cnode)) {
    *control_flow_scene = true;
    return;
  }

  const auto &inplace_node_abs = node->abstract();
  // Currently, only consider the case where the inplace operator has only
  // one output and one inplace input.
  if (!inplace_node_abs->isa<abstract::AbstractRefTensor>() || prim->rw_write_input_indexes().size() != 1) {
    MS_LOG(DEBUG) << "The inplace node is: " << node->DebugString();
    return;
  }
  const auto &rw_write_input = cnode->input(prim->rw_write_input_indexes()[0] + 1);
  const auto &rw_write_input_abs = rw_write_input->abstract();
  auto input_ref = rw_write_input_abs->cast<abstract::AbstractRefPtr>();
  MS_EXCEPTION_IF_NULL(input_ref);
  if (input_ref->is_view_output()) {
    auto view_op = input_ref->user_data<CNode>(kOriginalViewOp);
    if (view_op != nullptr) {
      inplace_node_abs->set_user_data<CNode>(kOriginalViewOp, view_op);
      MS_LOG(DEBUG) << "Mark view operator to inplace abstract, view_op: " << view_op->DebugString()
                    << " abstract: " << inplace_node_abs->ToString();
    }
  }
}

void MarkViewOpToAbstract(const FuncGraphPtr &func_graph, bool *control_flow_scene) {
  const auto &nodes = TopoSort(func_graph->get_return());
  for (size_t i = 0; i < nodes.size(); ++i) {
    MarkViewOp(nodes[i], control_flow_scene);
    if (*control_flow_scene) {
      return;
    }
  }
}
}  // namespace

void VirtualViewGradInsert(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  MS_EXCEPTION_IF_NULL(root);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // Insert VirtualViewGrad op for func_graph and sub_graphs
  VirtualViewGradInsertInner(root, manager);
  auto sub_graphs = root->func_graphs_used_total();
  for (const auto &sub_graph : sub_graphs) {
    VirtualViewGradInsertInner(sub_graph, manager);
  }

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("opt_insert_virtualviewgrad.ir", root);
  }
#endif
}

void RemoveRedundantVirtualViewGrad(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  MS_EXCEPTION_IF_NULL(root);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // Insert VirtualViewGrad op for func_graph and sub_graphs
  RemoveRedundantVirtualViewGradInner(root, manager);
  auto sub_graphs = root->func_graphs_used_total();
  for (const auto &sub_graph : sub_graphs) {
    RemoveRedundantVirtualViewGradInner(sub_graph, manager);
  }

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("opt_remove_redundant_virtualviewgrad.ir", root);
  }
#endif
}

// {prim::kPrimVirtualViewGrad, X, Y, ..., U} ==> X
AnfNodePtr VirtualViewGradEliminater::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimVirtualViewGrad) || node->func_graph() == nullptr) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  return cnode->input(kIndex1);
}

bool PreprocessForVirtualViewGradInsert(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  // mark view operator to abstract.
  bool control_flow_scene = false;
  MarkViewOpToAbstract(root, &control_flow_scene);
  if (control_flow_scene) {
    return true;
  }
  const auto &fg_used_total = root->func_graphs_used_total();
  for (const auto &fg : fg_used_total) {
    MarkViewOpToAbstract(fg, &control_flow_scene);
    if (control_flow_scene) {
      return true;
    }
  }
  return false;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
