/**
 * Copyright 2020-2025 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/ad/grad.h"
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/inplace_input_replace.h"
#include "frontend/operator/composite/composite.h"
#include "frontend/optimizer/irpass/check_invalid_view_inplace_dout.h"
#include "ir/func_graph_cloner.h"
#include "utils/ms_context.h"
#include "utils/symbolic.h"
#include "include/common/utils/parallel_context.h"

namespace mindspore {
namespace ad {
namespace {
constexpr auto kAlreadyCheck = "already_check";
constexpr auto kNeedGradFlag = "need_grad";
constexpr auto kHasViewOutputFlag = "has_view_output";
constexpr auto kCheckViewInplaceGradFlag = "view_inplace_grad_validate";
constexpr auto kSetNeedGradFlag = "set_need_grad_flag";

FuncGraphPtr PartialEliminateOptPass(const pipeline::ResourcePtr &resource, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(resource);

  opt::irpass::OptimizeIRPassLib irpass;
  opt::OptPassConfig partial_eliminate_opt_ = opt::OptPassConfig(
    {irpass.partial_eliminate_, irpass.switch_partial_eliminater_, irpass.switch_layer_partial_eliminater_});
  opt::OptPassGroupMap map({{"partial_eliminate_", partial_eliminate_opt_}});

  auto after_lift_opt = opt::Optimizer::MakeOptimizer("partial_eliminate", resource, map);

  FuncGraphPtr opt_fg = nullptr;
  ProfileExecute(MsProfile::GetProfile()->Step("partial_eliminate_before_grad"),
                 [&after_lift_opt, func_graph, &opt_fg]() { opt_fg = after_lift_opt->step(func_graph, true); });
  return opt_fg;
}

FuncGraphVector PartialEliminateMulti(const pipeline::ResourceBasePtr &resource, const FuncGraphVector &func_graphs) {
  auto new_res = std::dynamic_pointer_cast<pipeline::Resource>(resource);
  if (new_res == nullptr) {
    MS_LOG(INTERNAL_EXCEPTION) << "Parameter resources is not a pipeline::Resource";
  }
  FuncGraphVector opt_fgs;
  for (const auto &func_graph : func_graphs) {
    auto opt_fg = PartialEliminateOptPass(new_res, func_graph);
#ifdef ENABLE_DUMP_IR
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->CanDump(kIntroductory)) {
      DumpIR("after_opt_" + opt_fg->ToString() + ".ir", opt_fg);
    }
#endif
    opt_fgs.push_back(opt_fg);
  }
  return opt_fgs;
}

FuncGraphPtr LiftFv(const pipeline::ResourceBasePtr &resource, const FuncGraphPtr &func_graph) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool enable_save_graphs = context->CanDump(kIntroductory);
  if (enable_save_graphs) {
    DumpIR("before_lift_" + func_graph->ToString() + ".ir", func_graph);
  }
#endif
  FuncGraphPtr new_fg = LiftingClone(func_graph);
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("after_lift_" + new_fg->ToString() + ".ir", new_fg);
  }
#endif
  auto new_res = std::dynamic_pointer_cast<pipeline::Resource>(resource);
  if (new_res == nullptr) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, func_graph->return_node())
      << "Parameter resources is not a pipeline::Resource";
  }
  auto opt_fg = PartialEliminateOptPass(new_res, new_fg);
#ifdef ENABLE_DUMP_IR
  if (enable_save_graphs) {
    DumpIR("after_opt_" + opt_fg->ToString() + ".ir", opt_fg);
  }
#endif
  return opt_fg;
}

FuncGraphVector LiftFvMulti(const pipeline::ResourceBasePtr &resource, const FuncGraphVector &func_graphs) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    for (const auto &func_graph : func_graphs) {
      DumpIR("before_lift_" + func_graph->ToString() + ".ir", func_graph);
    }
  }
#endif
  bool has_used_fg = std::any_of(func_graphs.cbegin(), func_graphs.cend(), [](const FuncGraphPtr &func_graph) {
    return func_graph->func_graphs_used().size() != 0;
  });
  // All func_graphs being graded don't have used funcgraphs, no need to do lifting clone.
  if (!has_used_fg) {
    return func_graphs;
  }
  FuncGraphVector new_fgs = LiftingCloneMulti(func_graphs);
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    for (const auto &new_fg : new_fgs) {
      DumpIR("after_lift_" + new_fg->ToString() + ".ir", new_fg);
    }
  }
#endif
  return PartialEliminateMulti(resource, new_fgs);
}

bool ForwardInputsEqual(const AnfNodeWeakPtrList &first_inputs, const AnfNodeWeakPtrList &second_inputs) {
  if (first_inputs.size() != second_inputs.size()) {
    return false;
  }
  for (size_t i = 1; i < first_inputs.size(); ++i) {
    if (HasAbstractMonad(first_inputs[i].lock()) && HasAbstractMonad(second_inputs[i].lock())) {
      continue;
    }
    if (first_inputs[i].lock() != second_inputs[i].lock()) {
      return false;
    }
  }
  return true;
}

AnfNodePtr GetJUser(const FuncGraphManagerPtr &manager, const AnfNodePtr &j_node) {
  auto iter = manager->node_users().find(j_node);
  if (iter == manager->node_users().end()) {
    return nullptr;
  }
  auto users = iter->second;
  if (users.size() != 1) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, j_node) << "The size of J users should be 1, but got " << users.size();
  }
  return users.begin()->first;
}

void AddToManage(const pipeline::ResourceBasePtr &resources, const FuncGraphPtr &func_graph) {
  auto manager_ptr = resources->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  manager_ptr->AddFuncGraph(func_graph);
}

void CheckAbstractViewOutput(const AnfNodePtr &node) {
  const auto &abs = node->abstract();
  if (abs == nullptr) {
    return;
  }
  bool need_throw_exception = false;
  auto has_view_output = abs->user_data<bool>(kHasViewOutputFlag);
  if (has_view_output != nullptr && *has_view_output) {
    need_throw_exception = true;
  }
  if (abs->isa<abstract::AbstractRefTensor>()) {
    const auto ref = abs->cast<abstract::AbstractRefPtr>();
    if (ref->is_view_output()) {
      need_throw_exception = true;
    }
  }
  if (need_throw_exception) {
    MS_LOG(EXCEPTION) << "The current view inplace differentiation scenario is not supported. "
                         "The code location is as follows:\n"
                      << trace::GetDebugInfoStr(node->debug_info());
  }
}

void CheckOutputInner(const AnfNodePtr &node) {
  auto has_checked = node->user_data<bool>(kCheckViewInplaceGradFlag);
  if (has_checked != nullptr && *has_checked) {
    MS_LOG(DEBUG) << "The node has checked: " << node->DebugString();
    return;
  }
  node->set_user_data<bool>(kCheckViewInplaceGradFlag, std::make_shared<bool>(true));
  CheckAbstractViewOutput(node);

  if (!node->isa<CNode>() || IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  // call node
  if (IsValueNode<FuncGraph>(cnode->input(0))) {
    FuncGraphPtr sub_graph = GetValueNode<FuncGraphPtr>(cnode->input(0));
    MS_EXCEPTION_IF_NULL(sub_graph);
    auto sub_graph_out = sub_graph->output();
    return CheckOutputInner(sub_graph_out);
  }

  // call switch, check func_graph, do not check the input args.
  if (IsPrimitiveCNode(cnode->input(0), prim::kPrimSwitch)) {
    return CheckOutputInner(cnode->input(0));
  }

  // switch node
  if (IsPrimitiveCNode(cnode, prim::kPrimSwitch)) {
    constexpr size_t true_index = 2;
    constexpr size_t false_index = 3;
    auto true_func = GetValueNode<FuncGraphPtr>(cnode->input(true_index));
    MS_EXCEPTION_IF_NULL(true_func);
    auto true_func_out = true_func->output();
    CheckOutputInner(true_func_out);
    auto false_func = GetValueNode<FuncGraphPtr>(cnode->input(false_index));
    MS_EXCEPTION_IF_NULL(false_func);
    auto false_func_out = false_func->output();
    return CheckOutputInner(false_func_out);
  }

  if (IsPrimitiveCNode(cnode, prim::kPrimDepend)) {
    return CheckOutputInner(cnode->input(1));
  }
  const auto &inputs = cnode->inputs();
  for (auto input : inputs) {
    CheckOutputInner(input);
  }
}

void CheckViewInplaceOutput(const FuncGraphPtr &func_graph) {
  const auto &output = func_graph->output();
  MS_EXCEPTION_IF_NULL(output);
  auto output_abs = output->abstract();
  if (output_abs != nullptr && output_abs->isa<abstract::AbstractRefTensor>()) {
    auto ref = output_abs->cast<abstract::AbstractRefPtr>();
    if (ref->is_view_output()) {
      MS_LOG(EXCEPTION) << "The current view inplace differentiation scenario is not supported. "
                           "The code location is as follows:\n"
                        << trace::GetDebugInfoStr(output->debug_info());
    }
    if (ref->is_view_input()) {
      return;
    }
  }
  CheckOutputInner(output);
}

bool IsInplaceNode(const AnfNodePtr &node) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto prim = GetValueNode<PrimitivePtr>(node->cast<CNodePtr>()->input(0));
  return prim != nullptr && prim->inplace_prim();
}

bool UpdateStateUseOnly(const AnfNodePtr &node, const NodeUsersMap &node_user_map) {
  auto node_users_iter = node_user_map.find(node);
  if (node_users_iter == node_user_map.end()) {
    return false;
  }
  return std::all_of(node_users_iter->second.begin(), node_users_iter->second.end(),
                     [](const auto &pair) { return IsPrimitiveCNode(pair.first, prim::kPrimUpdateState); });
}

bool IsViewOutput(const AnfNodePtr &node) {
  auto abs = node->abstract();
  if (abs != nullptr && abs->isa<abstract::AbstractRefTensor>()) {
    const auto ref = abs->cast<abstract::AbstractRefPtr>();
    if (ref->is_view_output()) {
      return true;
    }
  }
  return false;
}

void GetNeedGradMapForUpdateStateUseOnlyNodes(const FuncGraphPtr &func_graph,
                                              std::map<AnfNodePtr, AnfNodePtr> *need_grad_map) {
  auto all_nodes = TopoSort(func_graph->get_return());
  const auto &mgr = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  const auto &node_users_map = mgr->node_users();

  for (const auto &node : all_nodes) {
    auto check_flag = node->user_data<bool>(kAlreadyCheck);
    auto already_check = check_flag != nullptr && *check_flag;
    if (!already_check && IsValueNode<FuncGraph>(node)) {
      FuncGraphPtr sub_graph = GetValueNode<FuncGraphPtr>(node);
      MS_EXCEPTION_IF_NULL(sub_graph);
      node->set_user_data<bool>(kAlreadyCheck, std::make_shared<bool>(true));
      GetNeedGradMapForUpdateStateUseOnlyNodes(sub_graph, need_grad_map);
      continue;
    }

    // is inplace node
    if (IsInplaceNode(node) && UpdateStateUseOnly(node, node_users_map)) {
      auto inplace_node = node->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(inplace_node);
      auto prim_value = inplace_node->input(0)->cast<ValueNodePtr>()->value();
      MS_EXCEPTION_IF_NULL(prim_value);
      auto prim = GetValue<PrimitivePtr>(prim_value);
      std::vector<size_t> rw_write_input_indexes = prim->rw_write_input_indexes();
      for (auto index : rw_write_input_indexes) {
        auto inplace_input = inplace_node->input(index + 1);
        if (IsViewOutput(inplace_input)) {
          (*need_grad_map)[inplace_input] = node;
        } else {
          (*need_grad_map)[inplace_node] = node;
        }
        node->set_user_data<bool>(kNeedGradFlag, std::make_shared<bool>(false));
      }
    }
  }
}

void SetFlagInner(const AnfNodePtr &node, const std::map<AnfNodePtr, AnfNodePtr> &need_grad_map) {
  auto already_set_flag = node->user_data<bool>(kSetNeedGradFlag);
  if (already_set_flag != nullptr && *already_set_flag) {
    MS_LOG(DEBUG) << "The node has checked: " << node->DebugString();
    return;
  }
  node->set_user_data<bool>(kSetNeedGradFlag, std::make_shared<bool>(true));
  auto iter = need_grad_map.find(node);
  if (need_grad_map.find(node) != need_grad_map.end()) {
    auto need_grad_node = iter->second;
    need_grad_node->set_user_data<bool>(kNeedGradFlag, std::make_shared<bool>(true));
  }
  if (!node->isa<CNode>()) {
    return;
  }
  auto cnode = node->cast<CNodePtr>();
  auto inputs = cnode->inputs();
  auto func_graph = cnode->func_graph();
  const auto &mgr = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mgr);
  const auto &node_users_map = mgr->node_users();

  for (auto input : inputs) {
    auto input_iter = need_grad_map.find(input);
    if (input_iter != need_grad_map.end()) {
      if (UpdateStateUseOnly(input, node_users_map)) {
        continue;
      }
      auto need_grad_node = input_iter->second;
      need_grad_node->set_user_data<bool>(kNeedGradFlag, std::make_shared<bool>(true));
    }
    if (input->isa<CNode>()) {
      SetFlagInner(input, need_grad_map);
    }
    if (IsValueNode<FuncGraph>(input)) {
      FuncGraphPtr sub_graph = GetValueNode<FuncGraphPtr>(input);
      MS_EXCEPTION_IF_NULL(sub_graph);
      auto sub_graph_out = sub_graph->output();
      SetFlagInner(sub_graph_out, need_grad_map);
    }
  }
}

void SetFlagForInplaceNodesUpdateStateUseOnly(const FuncGraphPtr &func_graph,
                                              const std::map<AnfNodePtr, AnfNodePtr> &need_grad_map) {
  const auto &output = func_graph->output();
  MS_EXCEPTION_IF_NULL(output);
  if (!output->isa<CNode>()) {
    return;
  }
  auto cnode = output->cast<CNodePtr>();
  SetFlagInner(cnode, need_grad_map);
}

bool NeedCheckInvalidViewInplaceDout(const std::string &scene) {
  //  1: Only check scenario 1
  //  2: Only check scenario 2
  //  Default(""): Check all invalid dout for view inplace scene
  //  Others: No invalid dout check for view inplace scene
  auto check_invalid_dout_level = common::GetCompileConfig("CHECK_INVALID_VIEW_INPLACE_DOUT_LEVEL");
  if (check_invalid_dout_level == "") {
    return true;
  }
  return check_invalid_dout_level == scene;
}

std::pair<CNodePtr, bool> IsCreatedByViewOp(const AnfNodePtr &node) {
  if (node->isa<CNode>()) {
    auto cnode = node->cast<CNodePtr>();
    auto prim = GetValueNode<PrimitivePtr>(cnode->input(0));
    if (prim != nullptr && prim->graph_view_prim()) {
      return {cnode, true};
    }
  }
  return {nullptr, IsViewOutput(node)};
}

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
}  // namespace

FuncGraphPtr GradOneFuncGraph(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer, bool is_top,
                              BpropAutoMonadLevel level, bool is_view_inplace) {
  MS_EXCEPTION_IF_NULL(func_graph);

  if (is_view_inplace && common::GetCompileConfig("ENABLE_VIEW_INPLACE_GRAD_NEW_METHOD") == "1") {
    // Insert VirtualViewGrad op for view+inplace scene
    MS_LOG(INFO) << "Insert VirtualViewGrad for func_graph: " << func_graph->ToString();
    (void)VirtualViewGradInsert(func_graph, optimizer);
    is_view_inplace = false;
  }

  // Do inplace input replacement
  mindspore::opt::DoInplaceInputReplace(func_graph, optimizer);

  if (is_view_inplace) {
    if (NeedCheckInvalidViewInplaceDout(opt::irpass::kCheckDoutLevelSceneTwo)) {
      CheckViewInplaceOutput(func_graph);
    }
    std::map<AnfNodePtr, AnfNodePtr> need_grad_map{};
    GetNeedGradMapForUpdateStateUseOnlyNodes(func_graph, &need_grad_map);
    SetFlagForInplaceNodesUpdateStateUseOnly(func_graph, need_grad_map);
  }
  auto gradkv = func_graph->transforms().find("grad");
  if (gradkv != func_graph->transforms().end()) {
    return gradkv->second.func_graph();
  }
  const auto &resources = optimizer->resource();
  AddToManage(resources, func_graph);
  auto multi_graph_sink = [&func_graph](const FuncGraphPtr &f) {
    if (MsContext::GetInstance()->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK)) {
      if (func_graph->has_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE)) {
        f->set_flag(FUNC_GRAPH_FLAG_IGNORE_VALUE, true);
      }
    }
  };

  auto f = std::make_shared<DFunctor>(func_graph, resources, is_top, is_view_inplace);
  auto user_defined = f->KUserDefined(func_graph);
  if (user_defined != nullptr) {
    multi_graph_sink(user_defined);
    if (is_top) {
      DFunctor::Clear();
    }
    return user_defined;
  }
  f->Init(is_top);
  f->MapObject();
  f->MapMorphism();
  f->Finish();
  auto res = f->k_graph();
  res->set_attr(kAttrBpropAutoMonadLevel, MakeValue<int>(level));
  auto tape = f->tape();
  tape->set_flag(mindspore::kFuncGraphFlagBackPropEntry, true);
  if (is_top) {
    DFunctor::Clear();
  }
  if (is_top && is_view_inplace) {
    auto get_real_bprop_out = std::make_shared<prim::GetRealBpropOut>("get_real_bprop_out");
    AnfNodePtr bout = tape->NewCNodeInOrder({NewValueNode(get_real_bprop_out), tape->output()});
    tape->set_output(bout);
  }

  // In the view + inplace scenario, ensure that the input corresponding to the inplace op that has not been updated in
  // place must not require gradient.
  if (is_view_inplace && NeedCheckInvalidViewInplaceDout(opt::irpass::kCheckDoutLevelSceneOne)) {
    mindspore::opt::irpass::MarkInvalidInplaceOpDout(res);
  }

  multi_graph_sink(res);
  (void)func_graph->transforms().emplace("grad", FuncGraphTransform(res));
  return res;
}

FuncGraphPtr Grad(const FuncGraphPtr &func_graph, const opt::OptimizerPtr &optimizer, bool is_top,
                  BpropAutoMonadLevel level, bool is_view_inplace) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto gradkv = func_graph->transforms().find("grad");
  if (gradkv != func_graph->transforms().end()) {
    return gradkv->second.func_graph();
  }
  if (!optimizer->is_first_order_j() && func_graph->has_attr(FUNC_GRAPH_ATTR_UNSUPPORT_HIGHER_GRAD_REASON)) {
    auto reason = func_graph->get_attr(FUNC_GRAPH_ATTR_UNSUPPORT_HIGHER_GRAD_REASON);
    MS_EXCEPTION_IF_NULL(reason);
    MS_EXCEPTION(NotSupportError) << "Higher-order differentiation is not supported for the current scenario, reason: "
                                  << GetValue<string>(reason);
  }

  const auto &resources = optimizer->resource();
  AddToManage(resources, func_graph);

  FuncGraphPtr grad_fg = func_graph;
  if (func_graph->func_graphs_used().size() != 0 && optimizer->is_first_order_j()) {
    lift_fv_before_grad = true;
    grad_fg = LiftFv(resources, func_graph);
  } else {
    lift_fv_before_grad = false;
  }
  return GradOneFuncGraph(grad_fg, optimizer, is_top, level, is_view_inplace);
}

FuncGraphVector GradMultiFuncGraph(const FuncGraphVector &func_graphs, const opt::OptimizerPtr &optimizer,
                                   const std::vector<bool> &is_view_inplace, bool is_top) {
  MS_EXCEPTION_IF_CHECK_FAIL(func_graphs.size() == is_view_inplace.size(), "GradMultiFuncGraph check size failed");
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);
  auto parallel_mode = parallel_context->parallel_mode();
  const bool is_parallel_mode =
    parallel_mode == parallel::kSemiAutoParallel || parallel_mode == parallel::kAutoParallel;
  BpropAutoMonadLevel bprop_auto_monad_level = is_parallel_mode ? kLevelTop : kLevelWhole;
  FuncGraphVector grad_fgs;
  if (func_graphs.size() == 1) {
    auto grad_fg = Grad(func_graphs[0], optimizer, is_top, bprop_auto_monad_level, is_view_inplace[0]);
    grad_fgs.push_back(grad_fg);
    return grad_fgs;
  }
  const auto &resources = optimizer->resource();
  auto manager_ptr = resources->manager();
  MS_EXCEPTION_IF_NULL(manager_ptr);
  for (const auto &func_graph : func_graphs) {
    manager_ptr->AddFuncGraph(func_graph);
  }
  FuncGraphVector before_grad_fgs;
  if (optimizer->is_first_order_j()) {
    lift_fv_before_grad = true;
    before_grad_fgs = LiftFvMulti(resources, func_graphs);
  } else {
    before_grad_fgs = func_graphs;
    lift_fv_before_grad = false;
  }
  for (size_t i = 0; i < before_grad_fgs.size(); ++i) {
    const auto &func_graph = before_grad_fgs[i];
    auto grad_fg = GradOneFuncGraph(func_graph, optimizer, is_top, bprop_auto_monad_level, is_view_inplace[i]);
    grad_fgs.push_back(grad_fg);
  }
  return grad_fgs;
}

FuncGraphPtr Kprim(const ValueNodePtr &value_node, const pipeline::ResourceBasePtr &resources) {
  auto fg = g_k_prims.KPrimitive(nullptr, value_node, resources, false);
  if (fg == nullptr) {
    return nullptr;
  }
  return BasicClone(fg);
}

MetaFuncGraphPtr Kmeta(const PrimitivePtr &prim, const pipeline::ResourceBasePtr &, const AnfNodePtr &node) {
  MetaFuncGraphPtr fg = g_k_prims.KMetaFuncGraph(prim, node);
  return fg;
}

void CleanRes() { DFunctor::Clear(); }

bool MergeForward(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::unordered_map<FuncGraphPtr, std::vector<AnfNodePtr>> forward_fg_to_j_nodes;
  auto all_nodes = TopoSort(root->get_return(), SuccDeeperSimple, AlwaysInclude);
  for (const auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimJ)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto merge_forward = cnode->user_data<bool>("merge_forward");
    if (merge_forward == nullptr || !(*merge_forward)) {
      continue;
    }
    auto forward_fg = GetValueNode<FuncGraphPtr>(cnode->input(1));
    if (forward_fg == nullptr) {
      continue;
    }
    (void)forward_fg_to_j_nodes[forward_fg].emplace_back(node);
  }
  bool change = false;
  for (const auto &iter : forward_fg_to_j_nodes) {
    auto &j_nodes = iter.second;
    MS_LOG(DEBUG) << "J nodes size is " << j_nodes.size();
    if (j_nodes.size() <= 1) {
      continue;
    }
    auto first_j_user = GetJUser(manager, j_nodes[0]);
    if (first_j_user == nullptr) {
      continue;
    }
    const auto &first_forward_inputs = first_j_user->cast<CNodePtr>()->weak_inputs();
    for (size_t i = 1; i < j_nodes.size(); ++i) {
      auto j_user = GetJUser(manager, j_nodes[i]);
      const auto &forward_inputs = j_user->cast<CNodePtr>()->weak_inputs();
      if (!ForwardInputsEqual(first_forward_inputs, forward_inputs)) {
        continue;
      }
      manager->Replace(j_user, first_j_user);
      MS_LOG(DEBUG) << "Replace J user " << j_user->DebugString() << " with the first J user "
                    << first_j_user->DebugString();
      change = true;
    }
  }
  return change;
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

void VirtualViewGradInsert(const FuncGraphPtr &root, const opt::OptimizerPtr &opt) {
  MS_EXCEPTION_IF_NULL(root);
  auto manager = opt->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // Insert VirtualViewGrad op for func_graph and sub_graphs
  VirtualViewGradInsertInner(root, manager);
  const auto &sub_graphs = root->func_graphs_used_total();
  for (auto sub_graph : sub_graphs) {
    VirtualViewGradInsertInner(sub_graph, manager);
  }

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("opt_insert_VirtualViewGrad_after.ir", root);
  }
#endif
}
}  // namespace ad
}  // namespace mindspore
