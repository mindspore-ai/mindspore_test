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

#include "frontend/optimizer/ad/dfunctor.h"

#include <algorithm>
#include <map>
#include <memory>
#include <string>

#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "ir/anf.h"
#include "utils/info.h"
#include "ir/func_graph_cloner.h"
#include "ir/manager.h"
#include "pipeline/jit/ps/resource.h"
#include "frontend/optimizer/ad/adjoint.h"
#include "frontend/operator/ops.h"
#include "utils/symbolic.h"
#include "utils/ms_context.h"
#include "pipeline/jit/ps/action.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/pynative/adapter.h"
#include "include/common/pynative/grad_state.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_h.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_o.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_u.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_z.h"
#include "mindspore/ccsrc/frontend/operator/composite/composite.h"

namespace mindspore {
namespace ad {
mindspore::HashMap<FuncGraphPtr, DFunctorPtr> DFunctor::func_graph_to_functor_;
mindspore::HashMap<AnfNodePtr, AdjointPtr> DFunctor::anfnode_to_adjoin_definition_;

bool lift_fv_before_grad = true;

namespace {
bool InplaceUsedByUpdateStateOnly(const AnfNodePtr &node, const NodeUsersMap &node_user_map) {
  if (!node->isa<CNode>()) {
    return false;
  }
  auto prim = GetValueNode<PrimitivePtr>(dyn_cast<CNode>(node)->input(0));
  if (prim == nullptr || !prim->inplace_prim()) {
    return false;
  }
  auto node_users_iter = node_user_map.find(node);
  if (node_users_iter == node_user_map.end()) {
    return false;
  }
  return std::all_of(node_users_iter->second.begin(), node_users_iter->second.end(),
                     [](const auto &pair) { return IsPrimitiveCNode(pair.first, prim::kPrimUpdateState); });
}

// Handle bprob of op which input dtype is real number and output dtype is complex number.
// If the dtype of a gradient(din) is complex number and the input of that is real number,
// only the real part of the gradient make sense in back propagate. We handle it by
// insert a Real() ops after the gradient in Eval stage. And firstly we mark the gradient nodes here.
// input: AnfNode with input of the forward op.
// din: CNodePtr with gradient of input.
void ComplexPreprocess(const AnfNodePtr &input, const CNodePtr &din) {
  MS_EXCEPTION_IF_NULL(input);
  TypePtr input_type = input->Type();
  if (input_type == nullptr || !input_type->isa<TensorType>()) {
    return;
  }
  input_type = input_type->cast_ptr<TensorType>()->element();
  MS_EXCEPTION_IF_NULL(input_type);
  if (input_type->type_id() == kNumberTypeComplex64 || input_type->type_id() == kNumberTypeComplex128) {
    return;
  }
  din->AddAttr(kAttrCheckComplex, MakeValue(true));
}

bool PrimitiveNeedSkip(const AnfNodePtr &node) {
  const auto prim = GetValueNode<PrimitivePtr>(node);
  if (!prim) {
    return false;
  }

  return IsPrimitiveEquals(prim, prim::kPrimReturn) || IsPrimitiveEquals(prim, prim::kPrimHookBackward) ||
         IsPrimitiveEquals(prim, prim::kPrimCellBackwardHook);
}

void CopyPrimitivePtrForFpropReplace(const FuncGraphPtr &primal_graph, const FuncGraphManagerPtr &manager) {
  MS_EXCEPTION_IF_NULL(primal_graph);
  MS_LOG(INFO) << "Copy primitive value node for fprop replace in gradjit function for fg: "
               << primal_graph->ToString();
  auto value_nodes = primal_graph->value_nodes();
  for (const auto &value_pair : value_nodes) {
    const auto &node = value_pair.first;
    MS_EXCEPTION_IF_NULL(node);
    if (!IsValueNode<Primitive>(node)) {
      continue;
    }
    if (IsPrimitive(node, prim::kPrimUpdateState) || PrimitiveNeedSkip(node)) {
      continue;
    }
    auto users = manager->node_users()[node];
    if (users.size() <= 1) {
      continue;
    }
    for (const auto &user : users) {
      auto new_value_node = NewValueNode(GetValueNode(node));
      auto cnode = user.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto index = user.second;
      (void)manager->SetEdge(cnode, index, new_value_node);
    }
  }
}

bool NeedGradForUpdateState(const CNodePtr &cnode, const NodeUsersMap &node_user_map, bool is_view_inplace) {
  if (!cnode->IsApply(prim::kPrimUpdateState) || !is_view_inplace) {
    return false;
  }
  return std::any_of(cnode->inputs().begin(), cnode->inputs().end(), [node_user_map](const auto &node_input) {
    return InplaceUsedByUpdateStateOnly(node_input, node_user_map);
  });
}

void DuplicateSideEffectNodes(const FuncGraphManagerPtr &manager, const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(manager);
  MS_EXCEPTION_IF_NULL(func_graph);

  auto value_nodes_origin = func_graph->value_nodes();
  for (const auto &value_pair : value_nodes_origin) {
    auto node = value_pair.first;
    auto prim = GetValueNode<PrimitivePtr>(node);
    if (prim == nullptr || !prim->HasAttr("side_effect_backprop_mem")) {
      continue;
    }
    auto users = manager->node_users()[node];
    if (users.size() <= 1) {
      continue;
    }
    for (const auto &user : users) {
      auto new_value_node = NewValueNode(prim);
      auto cnode = user.first->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(cnode);
      auto index = user.second;
      (void)manager->SetEdge(cnode, index, new_value_node);
    }
  }
}
}  // namespace

DFunctor::DFunctor(const FuncGraphPtr &primal_graph, const pipeline::ResourceBasePtr &resources, bool is_top,
                   bool is_view_inplace)
    : primal_graph_(primal_graph),
      resources_(resources),
      need_cut_(false),
      is_top_(is_top),
      is_view_inplace_(is_view_inplace) {
  {
    TraceGuard guard(MakeTraceInfo<TraceGradFprop>(primal_graph->debug_info()));
    k_graph_ = std::make_shared<FuncGraph>();
  }
  // To keep switch or switch_layer's inputs from being inlined
  k_graph_->set_indirect(primal_graph->indirect());
  k_graph_->set_stage(primal_graph->stage());
  k_graph_->set_segment(primal_graph->segment());

  {
    TraceGuard guard(MakeTraceInfo<TraceGradBprop>(primal_graph->debug_info()));
    tape_ = std::make_shared<FuncGraph>();
  }
  tape_->set_stage(primal_graph->stage());
  tape_->set_segment(primal_graph->segment());

  dout_ = tape_->add_parameter();
  if (is_view_inplace && is_top_) {
    auto get_dout_tuple = std::make_shared<prim::GenerateBpropOutTuple>("get_dout_tuple");
    dout_ = tape_->NewCNodeInOrder({NewValueNode(get_dout_tuple), dout_});
  }
}

void DFunctor::Init(bool is_top) {
  func_graph_to_functor_[primal_graph_] = shared_from_this();
  is_top_ = is_top;
}

void DFunctor::Finish() {
  CallDoutHoleOnTape();
  EliminatePrimalGraph();
}

void DFunctor::Clear() {
  func_graph_to_functor_.clear();
  anfnode_to_adjoin_definition_.clear();
}

void DFunctor::BackPropagateFv(const AnfNodePtr &fv, const AnfNodePtr &din) {
  MS_EXCEPTION_IF_NULL(fv);
  if (lift_fv_before_grad) {
    MS_EXCEPTION_IF_NULL(fv->func_graph());
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, fv)
      << "Can not find adjoint in anfnode_to_adjoin_ fv:" << fv->func_graph()->ToString() << " " << fv->ToString()
      << ".";
  }
  auto fv_adjoint = anfnode_to_adjoin_.find(fv);
  if (fv_adjoint == anfnode_to_adjoin_.end()) {
    MS_LOG(DEBUG) << "Can not find adjoint in anfnode_to_adjoin_ fv " << fv->func_graph()->ToString() << " "
                  << fv->ToString() << ".";

    if (fv->func_graph() == primal_graph_) {
      // If this fv is not mapped by MapMorphism because of cnode order, then map it now.
      (void)MapMorphism(fv);
      fv_adjoint = anfnode_to_adjoin_.find(fv);
      if (fv_adjoint == anfnode_to_adjoin_.end()) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, fv) << "Can not find adjoint in anfnode_to_adjoin_ fv "
                                                 << fv->func_graph()->ToString() << " " << fv->ToString() << ".";
      }
      if (is_view_inplace_ && fv->isa<CNode>()) {
        BackPropagate(dyn_cast<CNode>(fv), fv_adjoint->second);
        fv_adjoint->second->set_back_bproped(true);
      }
    } else {
      fv_adjoint = anfnode_to_adjoin_indirect_fv_.find(fv);
      if (fv_adjoint == anfnode_to_adjoin_indirect_fv_.end()) {
        MS_LOG(DEBUG) << "Can not find adjoint in anfnode_to_adjoin_indirect_fv_ fv " << fv->func_graph()->ToString()
                      << " " << fv->ToString() << ".";
        auto parent_adjoint = FindAdjoint(fv);
        AdjointPtr adjoint = nullptr;
        if (parent_adjoint != nullptr) {
          adjoint = std::make_shared<Adjoint>(fv, parent_adjoint->k(), tape_, is_view_inplace_);
        } else {
          MS_LOG(DEBUG) << "Can not find adjoint definition fv, add a k hole " << fv->func_graph()->ToString() << " "
                        << fv->ToString() << ".";
          adjoint = std::make_shared<Adjoint>(fv, nullptr, tape_, is_view_inplace_);
        }
        anfnode_to_adjoin_indirect_fv_[fv] = adjoint;
        fv_adjoint = anfnode_to_adjoin_indirect_fv_.find(fv);
      }
    }
  }
  auto fv_node = fv_adjoint->second->k();
  auto cached_envitem_iter = anfnode_to_envitem_.find(fv_node);
  CNodePtr embed_node;
  CNodePtr default_val_node;
  if (cached_envitem_iter != anfnode_to_envitem_.end()) {
    embed_node = cached_envitem_iter->second.first;
    default_val_node = cached_envitem_iter->second.second;
  } else {
    embed_node = tape_->NewCNode({NewValueNode(prim::kPrimEmbed), fv_node});
    default_val_node = tape_->NewCNode({NewValueNode(prim::GetPythonOps("zeros_like")), fv_node});
    fv_adjoint->second->RegisterKUser(embed_node, 1);
    fv_adjoint->second->RegisterKUser(default_val_node, 1);
    anfnode_to_envitem_[fv_node] = std::make_pair(embed_node, default_val_node);
  }
  auto dfv = tape_->NewCNode({NewValueNode(prim::kPrimEnvironGet), din, embed_node, default_val_node});
  MS_LOG(DEBUG) << "Find adjoint in anfnode_to_adjoin_ or anfnode_to_adjoin_indirect_fv_ fv "
                << fv->func_graph()->ToString() << " " << fv->ToString() << ".";
  MS_LOG(DEBUG) << "Get item from " << din->ToString() << " key " << embed_node->ToString() << ".";
  fv_adjoint->second->AccumulateDout(dfv);
}

void DFunctor::BackPropagateSwitchLayer(const CNodePtr &cnode_morph, const CNodePtr &env) {
  // Take switch_layer as a set of candidate functions.
  constexpr size_t input_tuple_index = 2;
  auto input = cnode_morph->input(input_tuple_index);
  if (!IsPrimitiveCNode(input, prim::kPrimMakeTuple)) {
    MS_LOG_WITH_NODE(EXCEPTION, cnode_morph)
      << "The 2th input of switch_layer expect a tuple of graphs, but got " << input->ToString() << ".";
  }
  mindspore::HashMap<AnfNodePtr, FuncGraphPtr> node_to_fg;
  auto tuple_graphs = input->cast_ptr<CNode>();
  MS_EXCEPTION_IF_NULL(tuple_graphs);
  for (size_t i = 1; i < tuple_graphs->size(); ++i) {
    auto graph = tuple_graphs->input(i);
    if (!IsValueNode<FuncGraph>(graph)) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_morph) << "The 2th input of switch_layer expect a tuple of graphs, but got "
                                               << graph->ToString() << " as the " << i << "th element.";
    }
    auto func_graph = GetValueNode<FuncGraphPtr>(graph);
    auto functor = func_graph_to_functor_.find(func_graph);
    if (functor == func_graph_to_functor_.end()) {
      MS_LOG_WITH_NODE(EXCEPTION, cnode_morph)
        << "Failed functor for subgraph does not exist input[" << i << "] " << func_graph->ToString() << ".";
    }
    // Consider direct and indirect fvs.
    for (auto fv : func_graph->free_variables_nodes()) {
      if (node_to_fg.find(fv) != node_to_fg.end()) {
        continue;
      }
      node_to_fg[fv] = func_graph;
      BackPropagateFv(fv, env);
    }
    for (auto indirect_fv : functor->second->anfnode_to_adjoin_indirect_fv_) {
      MS_LOG(DEBUG) << "Backprop indirect fv " << func_graph->ToString() << " " << indirect_fv.first->ToString() << ".";
      if (node_to_fg.find(indirect_fv.first) != node_to_fg.end()) {
        continue;
      }
      node_to_fg[indirect_fv.first] = func_graph;
      BackPropagateFv(indirect_fv.first, env);
    }
  }
}

static bool HasSideEffectBackProp(const CNodePtr &cnode) {
  if (IsPrimitiveCNode(cnode)) {
    const auto &prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    auto bprop_flag = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP);
    return bprop_flag;
  }
  return false;
}

static bool HasSideEffectBackPropMem(const CNodePtr &cnode) {
  if (IsPrimitiveCNode(cnode)) {
    const auto &prim = GetCNodePrimitive(cnode);
    MS_EXCEPTION_IF_NULL(prim);
    auto bprop_flag = GetPrimitiveFlag(prim, GRAPH_FLAG_SIDE_EFFECT_BACKPROP_MEM);
    return bprop_flag;
  }
  return false;
}

static AnfNodePtr SkipHookNodeInBackProp(const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (IsPrimitiveCNode(node, prim::kPrimHookBackward) || IsPrimitiveCNode(node, prim::kPrimCellBackwardHook)) {
    MS_LOG(WARNING) << "Hook operation does not work in graph mode or functions decorated with 'jit', it will be "
                       "eliminated during compilation.";
    auto output_cnode = node->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(output_cnode);
    if (output_cnode->size() - 1 == 1) {
      return output_cnode->input(1);
    }
    // Replace hook node with make tuple node.
    abstract::AbstractBasePtrList multi_output_abs;
    std::vector<AnfNodePtr> multi_output_nodes{NewValueNode(prim::kPrimMakeTuple)};
    (void)std::for_each(output_cnode->weak_inputs().cbegin() + 1, output_cnode->weak_inputs().cend(),
                        [&multi_output_nodes, &multi_output_abs](const AnfNodeWeakPtr &weak_inp) {
                          AnfNodePtr inp = weak_inp.lock();
                          MS_EXCEPTION_IF_NULL(inp);
                          (void)multi_output_nodes.emplace_back(inp);
                          (void)multi_output_abs.emplace_back(inp->abstract());
                        });
    auto primal_graph = node->func_graph();
    MS_EXCEPTION_IF_NULL(primal_graph);
    auto make_tuple = primal_graph->NewCNode(std::move(multi_output_nodes));
    make_tuple->set_abstract(std::make_shared<abstract::AbstractTuple>(multi_output_abs));
    auto mng = primal_graph->manager();
    MS_EXCEPTION_IF_NULL(mng);
    if (!mng->Replace(node, make_tuple)) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node)
        << "Failed to replace old node: " << node->DebugString() << " with new node: " << make_tuple->DebugString();
    }
    return make_tuple;
  }
  if (IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
    auto tuple_get_item = node->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(tuple_get_item);
    auto inp = tuple_get_item->input(1);
    if (IsPrimitiveCNode(inp, prim::kPrimHookBackward) || IsPrimitiveCNode(inp, prim::kPrimCellBackwardHook)) {
      MS_LOG(WARNING) << "Hook operation does not work in graph mode or functions decorated with 'jit', it will be "
                         "eliminated during compilation.";
      constexpr size_t idx = 2;
      auto v_node = dyn_cast_ptr<ValueNode>(tuple_get_item->input(idx));
      MS_EXCEPTION_IF_NULL(v_node);
      auto out_idx = GetValue<int64_t>(v_node->value());
      auto cnode = inp->cast_ptr<CNode>();
      MS_EXCEPTION_IF_NULL(cnode);
      return cnode->input(LongToSize(out_idx) + 1);
    }
  }
  return node;
}

bool IsLastNodeOfGraph(const AnfNodePtr &node, const NodeUsersMap &node_users_map) {
  auto node_user_iter = node_users_map.find(node);
  if (node_user_iter == node_users_map.end()) {
    return false;
  }
  return std::all_of(node_user_iter->second.begin(), node_user_iter->second.end(),
                     [](const auto &pair) { return IsPrimitiveCNode(pair.first, prim::kPrimReturn); });
}

CNodePtr DFunctor::CalculateDoutTuple(const CNodePtr &cnode_morph, const CNodePtr &din_tuple,
                                      const AdjointPtr &node_adjoint, int index) {
  bool single_tensor_view = false;
  bool inplace_prim = false;
  auto prim = GetValueNode<PrimitivePtr>(cnode_morph->input(0));
  if (prim != nullptr) {
    const auto &prim_name = prim->name();
    inplace_prim = prim->inplace_prim();
    auto op_def = mindspore::ops::GetOpDef(prim_name);
    if (op_def != nullptr) {
      single_tensor_view =
        op_def->is_graph_view_ && (op_def->returns_[0].arg_dtype_ == mindspore::ops::OP_DTYPE::DT_TENSOR);
    }
  }

  auto caller = node_adjoint->caller();
  const auto &node_users_map = resources_->manager()->node_users();
  // For Some ops of Framework:
  if (IsPrimitiveCNode(cnode_morph, prim::kPrimDepend) && (index == 1)) {
    if (IsLastNodeOfGraph(cnode_morph, node_users_map)) {
      auto get_depend_dout_tuple = std::make_shared<prim::GetDependDoutTuple>("get_depend_dout_tuple");
      return caller->NewCNodeInOrder({NewValueNode(get_depend_dout_tuple), din_tuple, dout_});
    }
    return dyn_cast<CNode>(node_adjoint->real_dout());
  }

  if (IsPrimitiveCNode(cnode_morph, prim::kPrimMakeTuple)) {
    if (IsLastNodeOfGraph(cnode_morph, node_users_map)) {
      return caller->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), dout_, NewValueNode(int64_t(index - 1))});
    }
    return caller->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), node_adjoint->real_dout(), NewValueNode(int64_t(index - 1))});
  }

  auto k_app = node_adjoint->k_app();
  if (k_app == nullptr) {
    return din_tuple;
  }

  if (IsPrimitiveCNode(cnode_morph, prim::kPrimTupleGetItem) && (index == 1)) {
    constexpr size_t input_idx = 1;
    constexpr size_t index_idx = 2;
    auto dout_temp = caller->NewCNodeInOrder({NewValueNode(prim::GetPythonOps("zeros_like")), k_app->input(input_idx)});
    auto generate_dout_tuple = std::make_shared<prim::GenerateBpropOutTuple>("generate_dout_tuple");
    generate_dout_tuple->set_ops_type(prim::OpsType::Type_Variable);
    auto dout_tuple_tmp = caller->NewCNodeInOrder({NewValueNode(generate_dout_tuple), dout_temp});
    return caller->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleSetItem), dout_tuple_tmp, k_app->input(index_idx), node_adjoint->real_dout()});
  }

  // Get Din/dmask/ops_type from din_tuple: (din, (dmask, ops_tye));
  auto din = caller->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), din_tuple, NewValueNode(int64_t(0))});

  if (inplace_prim) {
    // For inplace_prim, Change the ops_type when do backpropagate.
    auto inplace_indexes = prim->inplace_input_indexes();
    auto iter = std::find(inplace_indexes.begin(), inplace_indexes.end(), (index - 1));
    if (iter != inplace_indexes.end()) {
      auto generate_inplace_dout_tuple = std::make_shared<prim::GenerateBpropOutTuple>("generate_inplace_dout_tuple");
      generate_inplace_dout_tuple->set_ops_type(prim::OpsType::Type_Inplace);
      return caller->NewCNodeInOrder({NewValueNode(generate_inplace_dout_tuple), din});
    }
    return din_tuple;
  }

  // For View_ops, Just record the first input.
  if (single_tensor_view && index == 1) {
    // Get Din/dmask/ops_type from node_adjoint->dout(): (din, (dmask, ops_tye));
    auto node_dout_tuple = caller->NewCNodeInOrder(
      {NewValueNode(prim::kPrimTupleGetItem), node_adjoint->real_dout(), NewValueNode(int64_t(1))});
    auto node_mask =
      caller->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), node_dout_tuple, NewValueNode(int64_t(0))});
    auto node_ops_type =
      caller->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), node_dout_tuple, NewValueNode(int64_t(1))});

    // Firstly, Initialize a dout_mask whose values are all `false`
    auto ori_mask =
      caller->NewCNodeInOrder({NewValueNode(prim::kPrimZerosLikeExt), din, NewValueNode(int64_t(kBool->type_id()))});

    constexpr size_t input_begin_index = 2;
    AnfNodePtrList viewed_mask_nodes{NewValueNode(prim), ori_mask};
    std::copy(k_app->inputs().begin() + input_begin_index, k_app->inputs().end() - 1,
              std::back_inserter(viewed_mask_nodes));
    auto mask_viewed = caller->NewCNodeInOrder(viewed_mask_nodes);
    auto mask_viewed_true = caller->NewCNodeInOrder(
      {NewValueNode(prim::kPrimOnesLikeExt), mask_viewed, NewValueNode(MakeValue<int64_t>(kBool->type_id()))});

    // Get view part by the view_ops and set the value to `true'.
    auto temp = caller->NewCNodeInOrder({NewValueNode(prim::kPrimInplaceCopy), mask_viewed, mask_viewed_true});
    ori_mask = caller->NewCNodeInOrder({NewValueNode(prim::kPrimDepend), ori_mask, temp});
    // Finally, Do masked_scatter based on dout_mask and the postback dout_mask.
    auto dout_mask = caller->NewCNodeInOrder({NewValueNode(prim::kPrimMaskedScatter), ori_mask, ori_mask, node_mask});
    auto dout_tuple = caller->NewCNodeInOrder(
      {NewValueNode(prim::kPrimMakeTuple), din,
       caller->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), dout_mask, node_ops_type})});
    return dout_tuple;
  }
  return din_tuple;
}

void DFunctor::AccumulateInputGradients(const CNodePtr &cnode_morph, const AdjointPtr &node_adjoint,
                                        const CNodePtr bprop_app) {
  for (size_t i = 0; i < cnode_morph->size(); i++) {
    auto din = tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_app, NewValueNode(SizeToLong(i))});
    auto input = SkipHookNodeInBackProp(cnode_morph->input(i));
    ComplexPreprocess(input, din);
    // Backprop sens wrt fvs.
    if (IsValueNode<FuncGraph>(input)) {
      auto func_graph = GetValueNode<FuncGraphPtr>(input);
      auto functor = func_graph_to_functor_.find(func_graph);
      if (functor == func_graph_to_functor_.end()) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode_morph)
          << "Failed functor for subgraph does not exist input[" << i << "] " << func_graph->ToString() << ".";
      }
      // Consider direct and indirect fvs.
      for (auto fv : func_graph->free_variables_nodes()) {
        BackPropagateFv(fv, din);
      }
      for (auto indirect_fv : functor->second->anfnode_to_adjoin_indirect_fv_) {
        MS_LOG(DEBUG) << "Backprop indirect fv " << func_graph->ToString() << ", " << indirect_fv.first->ToString()
                      << ".";
        BackPropagateFv(indirect_fv.first, din);
      }
      continue;
    }
    // Backprop sens wrt inputs.
    auto input_adjoint = anfnode_to_adjoin_.find(input);
    if (input_adjoint == anfnode_to_adjoin_.end()) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, input)
        << "The adjoint does not exist input[" << i << "] " << input->ToString()
        << ". primal_graph_: " << primal_graph_->ToString();
    }
    if (!is_view_inplace_) {
      input_adjoint->second->AccumulateDout(din);
      continue;
    }
    auto node_input = cnode_morph->input(i);
    const auto &node_users_map = resources_->manager()->node_users();
    constexpr auto need_grad_flag = "need_grad";
    bool need_grad = node_input->has_user_data(need_grad_flag) && *node_input->user_data<bool>(need_grad_flag);
    if (InplaceUsedByUpdateStateOnly(node_input, node_users_map) && need_grad) {
      // Initialize a dout for the cnode used only by updatestate.
      MS_LOG(DEBUG) << "The Inplace node only used by UpdateState needs to pass the gradient. The node is:"
                    << node_input->DebugString();
      auto caller = input_adjoint->second->caller();
      auto real_din = caller->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), din, NewValueNode(int64_t(0))});
      auto dmask_tuple =
        caller->NewCNodeInOrder({NewValueNode(prim::kPrimTupleGetItem), din, NewValueNode(int64_t(1))});
      auto din_ones = input_adjoint->second->caller()->NewCNodeInOrder({NewValueNode(prim::kPrimOnesLike), real_din});
      auto din_ones_tuple = caller->NewCNodeInOrder({NewValueNode(prim::kPrimMakeTuple), din_ones, dmask_tuple});
      input_adjoint->second->AccumulateDout(din_ones_tuple);
    }
    din = CalculateDoutTuple(cnode_morph, din, node_adjoint, i);
    input_adjoint->second->AccumulateDout(din);
  }
}

void DFunctor::BackPropagate(const CNodePtr &cnode_morph, const AdjointPtr &node_adjoint) {
  // Do BackPropagate When node_adjoint has not BackPropagated.
  if (cnode_morph->stop_gradient() || node_adjoint->back_bproped()) {
    return;
  }
  auto bprop = k_graph_->NewCNode(
    {NewValueNode(prim::kPrimTupleGetItem), node_adjoint->k_app(), NewValueNode(static_cast<int64_t>(1))});
  // Call with delimited continuation dout.
  CNodePtr bprop_app;
  if (HasSideEffectBackProp(cnode_morph)) {
    if (is_view_inplace_) {
      bprop_app = tape_->NewCNodeInOrder({bprop, node_adjoint->dout()});
    } else {
      bprop_app = tape_->NewCNodeInFront({bprop, node_adjoint->dout()});
    }
    tape_->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);
  } else {
    if (common::GetCompileConfig("PUT_ALL_CNODE_INTO_ORDER_LIST") == "0") {
      bprop_app = tape_->NewCNode({bprop, node_adjoint->dout()});
    } else {
      bprop_app = tape_->NewCNodeInOrder({bprop, node_adjoint->dout()});
    }
  }

  if (HasSideEffectBackPropMem(cnode_morph)) {
    bprop_app->AddAttr(kAttrSideEffectBpropApp, MakeValue(true));
    k_graph_->set_flag(kAttrSideEffectBpropAppPropagate, true);
  }
  if (node_adjoint->side_effect_bprop_app_propagate()) {
    bprop_app->AddAttr(kAttrSideEffectBpropAppPropagate, MakeValue(true));
    k_graph_->set_flag(kAttrSideEffectBpropAppPropagate, true);
  }
  node_adjoint->RegisterDoutUser(bprop_app, 1);
  // Special case for switch_layer
  if (IsPrimitiveCNode(cnode_morph, prim::kPrimSwitchLayer)) {
    auto din =
      tape_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), bprop_app, NewValueNode(static_cast<int64_t>(0))});
    BackPropagateSwitchLayer(cnode_morph, din);
    return;
  }

  AccumulateInputGradients(cnode_morph, node_adjoint, bprop_app);
}

// Map a morphism.
AdjointPtr DFunctor::MapMorphism(const AnfNodePtr &morph) {
  constexpr int recursive_level = 4;
  MS_LOG(DEBUG) << "Start: " << morph->DebugString(recursive_level);
  // MapMorphism All type except CNode should already be mapped by MapObject.
  if (!morph->isa<CNode>()) {
    return nullptr;
  }
  // for free variable, which may be handled in MapValueObject, just return it
  auto node_adjoint_found = anfnode_to_adjoin_.find(morph);
  if (node_adjoint_found != anfnode_to_adjoin_.end()) {
    return node_adjoint_found->second;
  }
  ScopeGuard scope_guard(morph->scope());
  auto cnode_morph = morph->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode_morph);

  std::vector<AnfNodePtr> inputs;
  std::vector<AdjointPtr> param_adjoints;
  bool side_effect_bprop_app_propagate = false;
  for (size_t i = 0; i < cnode_morph->size(); i++) {
    auto node = SkipHookNodeInBackProp(cnode_morph->input(i));
    AdjointPtr node_adjoint = nullptr;
    auto node_adjoint_iter = anfnode_to_adjoin_.find(node);
    if (node_adjoint_iter != anfnode_to_adjoin_.end()) {
      node_adjoint = node_adjoint_iter->second;
    } else {
      // Input might be a CNode that needs to be handled previously.
      node_adjoint = MapMorphism(node);
    }
    if (node_adjoint == nullptr) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node) << "The node adjoint is null, " << node->DebugString();
    }
    AnfNodePtr k = node_adjoint->k();
    if (k == nullptr) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node)
        << "The adjoint node does not exist, input[" << i << "] " << node->ToString() << ".";
    }
    if (i == 0) {
      auto k_fg = GetValueNode<FuncGraphPtr>(k);
      if (k_fg != nullptr) {
        (void)k_fg->transforms().emplace("primal_cnode", FuncGraphTransform(cnode_morph));
        side_effect_bprop_app_propagate = k_fg->has_flag(kAttrSideEffectBpropAppPropagate);
      }
    }
    inputs.push_back(k);
    param_adjoints.push_back(node_adjoint);
  }
  CNodePtr k_app = nullptr;
  {
    TraceGuard guard(MakeTraceInfo<TraceGradFpropApp>(cnode_morph->debug_info()));
    k_app = k_graph_->NewCNode(inputs);
    const DebugInfoPtr &old_debug_info = cnode_morph->debug_info();
    if (old_debug_info != nullptr) {
      const auto &old_real_loc = old_debug_info->real_loc();
      if (!old_real_loc.empty() && k_app->debug_info() != nullptr) {
        k_app->debug_info()->set_real_loc(old_real_loc);
      }
    }
  }

  for (size_t i = 0; i < param_adjoints.size(); ++i) {
    param_adjoints[i]->RegisterKUser(k_app, i);
  }
  // Do forward computation
  auto forward_app =
    k_graph_->NewCNode({NewValueNode(prim::kPrimTupleGetItem), k_app, NewValueNode(static_cast<int64_t>(0))});
  // K:: cnode -> forward_app
  auto node_adjoint = std::make_shared<Adjoint>(morph, forward_app, tape_, is_view_inplace_);
  node_adjoint->set_k_app(k_app);
  node_adjoint->set_side_effect_bprop_app_propagate(side_effect_bprop_app_propagate);
  UpdateAdjoint(node_adjoint);
  anfnode_to_adjoin_[morph] = node_adjoint;
  MS_LOG(DEBUG) << "End, node: " << morph->DebugString(recursive_level);

  if (!is_view_inplace_) {
    // Do sens backpropagation.
    BackPropagate(cnode_morph, node_adjoint);
  }
  return node_adjoint;
}

bool DFunctor::IsFreeMorphism(const AnfNodePtr &node) {
  // Do not care about non-CNode
  if (!node->isa<CNode>()) {
    return false;
  }
  // Do not care about kPrimReturn
  if (IsPrimitiveCNode(node, prim::kPrimReturn)) {
    return false;
  }
  MS_EXCEPTION_IF_NULL(primal_graph_->manager());
  auto &node_users = primal_graph_->manager()->node_users();
  auto iter = node_users.find(node);
  if (iter == node_users.end()) {
    return false;
  }
  auto &users = iter->second;
  // Do not care about isolated morphisms
  if (users.empty()) {
    return false;
  }
  // Not free if it's used by some node in primal_graph
  bool nonfree = std::any_of(std::begin(users), std::end(users), [&](const auto &kv) {
    auto &user = kv.first;
    return user->func_graph() == primal_graph_;
  });
  return !nonfree;
}

void DFunctor::MapFreeMorphism() {
  // Handle cnode not attached to output, that might be referred in other functions.
  for (auto &node : primal_graph_->nodes()) {
    if (!IsFreeMorphism(node)) {
      continue;
    }
    MS_LOG(DEBUG) << "Map nonoutput cnode after MapMorphism " << node->ToString() << ".";
    (void)MapMorphism(node);

    if (is_view_inplace_) {
      auto cnode = dyn_cast<CNode>(node);
      auto node_adjoint_iter = anfnode_to_adjoin_.find(node);
      if (node_adjoint_iter == anfnode_to_adjoin_.end()) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node) << "The node adjoint is null, " << node->DebugString();
      }
      auto node_adjoint = node_adjoint_iter->second;
      BackPropagate(cnode, node_adjoint);
      node_adjoint->set_back_bproped(true);
    }
  }
}

AnfNodePtr DFunctor::AttachFvDoutToTape(const AnfNodePtr &grad_fv) {
  AnfNodePtr new_grad_fv = grad_fv;
  // Add grads wrt fv.
  const auto &free_variables_nodes = primal_graph_->free_variables_nodes();
  if (!is_top_ && free_variables_nodes.size() != 0) {
    if (lift_fv_before_grad) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, grad_fv)
        << "The direct fv size is: " << free_variables_nodes.size() << " in " << primal_graph_->ToString() << ".";
    }
  }

  for (auto &fv : free_variables_nodes) {
    if (IsPrimitiveCNode(fv, prim::kPrimJ)) {  // Ignore if FV is a J CNode.
      continue;
    }
    auto fv_adjoint = anfnode_to_adjoin_.find(fv);
    if (fv_adjoint == anfnode_to_adjoin_.end()) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, fv) << "The fv adjoint does not exist " << fv->ToString() << ".";
    }
    auto node = tape_->NewCNode({NewValueNode(prim::kPrimEmbed), fv_adjoint->second->k()});
    fv_adjoint->second->RegisterKUser(node, 1);
    auto sens = fv_adjoint->second->dout();
    new_grad_fv = tape_->NewCNode({NewValueNode(prim::kPrimEnvironSet), new_grad_fv, node, sens});
    constexpr size_t sens_index = 3;
    fv_adjoint->second->RegisterDoutUser(new_grad_fv->cast<CNodePtr>(), sens_index);
    MS_LOG(DEBUG) << "Add fv sens " << sens->ToString() << " to " << new_grad_fv->ToString() << " " << fv->ToString()
                  << " " << primal_graph_->ToString() << ".";
  }
  return new_grad_fv;
}

AnfNodePtr DFunctor::AttachIndirectFvDoutToTape(const AnfNodePtr &grad_fv) {
  if (lift_fv_before_grad) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, grad_fv) << "Lift free variable case: backprop indirect fv "
                                                  << grad_fv->ToString() << " " << primal_graph_->ToString() << ".";
  }
  AnfNodePtr new_grad_fv = grad_fv;
  // Add indirect fv bprop.
  for (auto &fv_adjoint : anfnode_to_adjoin_indirect_fv_) {
    MS_LOG(DEBUG) << "Backprop indirect fv " << fv_adjoint.first->ToString() << " " << primal_graph_->ToString() << ".";
    auto node = tape_->NewCNode({NewValueNode(prim::kPrimEmbed), fv_adjoint.second->k()});
    fv_adjoint.second->RegisterKUser(node, 1);
    auto sens = fv_adjoint.second->dout();
    new_grad_fv = tape_->NewCNode({NewValueNode(prim::kPrimEnvironSet), new_grad_fv, node, sens});
    constexpr size_t sens_index = 3;
    fv_adjoint.second->RegisterDoutUser(new_grad_fv->cast<CNodePtr>(), sens_index);
    MS_LOG(DEBUG) << "Add indirect fv sens " << sens->ToString() << " to " << new_grad_fv->ToString() << ".";
  }
  return new_grad_fv;
}

void DFunctor::MapMorphism() {
  // Set stop_gradient before MapMorphism.
  BroadCastStopFlag();

  // Handle free morphism before output, because in some case, free morphism might depend on output's fv tangent
  MapFreeMorphism();
  // Skip HookBackward op and CellBackwardHook op when it is the output node.
  auto output_node = primal_graph_->output();
  output_node = SkipHookNodeInBackProp(output_node);
  // Handle morphism from output.
  // Topo sort all nodes firstly in case of stack overflow fault.
  auto nodes = TopoSort(output_node, SuccIncoming, [this](const AnfNodePtr &node) -> IncludeType {
    MS_EXCEPTION_IF_NULL(node);
    if (node->func_graph() == nullptr || node->func_graph() != primal_graph_ || node->isa<Parameter>()) {
      return EXCLUDE;
    }
    return FOLLOW;
  });
  for (const auto &node : nodes) {
    (void)MapMorphism(SkipHookNodeInBackProp(node));
  }

  if (is_view_inplace_) {
    // Do backPropagate by reversed order for view_inplace grad.
    std::reverse(nodes.begin(), nodes.end());
    for (const auto &pre_node : nodes) {
      auto node = SkipHookNodeInBackProp(pre_node);
      if (!node->isa<CNode>()) {
        continue;
      }
      auto cnode = dyn_cast<CNode>(node);
      auto node_adjoint_iter = anfnode_to_adjoin_.find(node);
      if (node_adjoint_iter == anfnode_to_adjoin_.end()) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node) << "The node adjoint is null, " << node->DebugString();
      }
      auto node_adjoint = node_adjoint_iter->second;
      BackPropagate(cnode, node_adjoint);
      node_adjoint->set_back_bproped(true);
    }
  }
  // Construct K for primal_graph_.
  auto output_adjoint = anfnode_to_adjoin_.find(output_node);
  // Attach dout_ parameter to output_adjoint.
  output_adjoint->second->AccumulateDout(dout_);

  // Set output for tape closure.
  AnfNodePtr grad_fv;
  if (lift_fv_before_grad) {
    grad_fv = AttachFvDoutToTape(NewEnviron(tape_));
  } else {
    grad_fv = AttachIndirectFvDoutToTape(AttachFvDoutToTape(NewEnviron(tape_)));
  }

  std::vector<AnfNodePtr> inputs{NewValueNode(prim::kPrimMakeTuple), grad_fv};
  // Add grads wrt inputs.
  std::vector<AdjointPtr> param_adjoints;
  for (auto &param : primal_graph_->parameters()) {
    auto param_adjoint = anfnode_to_adjoin_.find(param);
    inputs.push_back(param_adjoint->second->dout());
    param_adjoints.push_back(param_adjoint->second);
  }
  auto tape_output = tape_->NewCNode(inputs);
  constexpr size_t offset_num = 2;
  for (size_t i = 0; i < param_adjoints.size(); ++i) {
    param_adjoints[i]->RegisterDoutUser(tape_output, i + offset_num);
  }
  tape_->set_output(tape_output);
  // Set output for k_graph_, K:: cnode->forward_app.
  auto forward_app = output_adjoint->second->k();
  auto output = k_graph_->NewCNode({NewValueNode(prim::kPrimMakeTuple), forward_app, NewValueNode(tape_)});
  output_adjoint->second->RegisterKUser(output, 1);
  k_graph_->set_output(output);
  (void)primal_graph_->transforms().emplace("grad", FuncGraphTransform(k_graph_));
  (void)k_graph_->transforms().emplace("primal", FuncGraphTransform(primal_graph_));
}

FuncGraphPtr DFunctor::KUserDefined(const FuncGraphPtr &primal) {
  // K user defined cell bprop.
  auto bprop = primal->transforms().find("bprop");
  if (bprop != primal->transforms().end()) {
    FuncGraphPtr bprop_graph = bprop->second.func_graph();
    resources_->manager()->AddFuncGraph(bprop_graph);

    (void)parse::ResolveFuncGraph(bprop_graph, resources_);
    if (!bprop_graph->free_variables_nodes().empty()) {
      MS_LOG_WITH_NODE(EXCEPTION, bprop_graph->return_node())
        << "The user defined 'bprop' function in scope " << primal->output()->scope()->name()
        << " does not support using Parameter.\n"
        << trace::GetDebugInfoStr(bprop_graph->debug_info());
    }
    // Check the func decorated by @custom_vjp.
    if (g_k_prims.CheckCustomVjp(bprop_graph)) {
      bprop_graph = g_k_prims.GetCustomVjpBprop(bprop_graph);
      bprop->second = FuncGraphTransform(bprop_graph);
    }

    bprop_graph->set_flag(mindspore::kFuncGraphFlagBackPropEntry, true);
    bprop_graph->set_flag(mindspore::kFuncGraphFlagReAutoMonad, true);

    auto fg = g_k_prims.KUserDefinedCellBprop(bprop_graph, primal, is_view_inplace_);
    if (fg == nullptr) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, primal->return_node())
        << "Failed to expand user defined Cell bprop " << primal->ToString() << " in scope "
        << primal->output()->scope()->name() << ".";
    }

    // Cache the grad func
    (void)primal->transforms().emplace("grad", FuncGraphTransform(fg));
    (void)fg->transforms().emplace("primal", FuncGraphTransform(primal));
    // Reset defer_inline to enable successive inlining
    primal->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, false);

    auto functor = std::make_shared<DFunctor>(primal, resources_, false, is_view_inplace_);
    functor->Init();
    functor->k_graph_ = fg;

    return fg;
  }
  return nullptr;
}

bool StopGradientForScalar(const CNodePtr &cnode) {
  auto grad_for_scalar = (MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) ||
                          common::GetCompileConfig("GRAD_FOR_SCALAR") == "1");
  if (grad_for_scalar) {
    return false;
  }
  auto abs = cnode->abstract();
  return abs != nullptr && abs->isa<abstract::AbstractScalar>();
}

// Construct representation graph for {CNode, Index} of Primitive.
AnfNodePtr DFunctor::MapPrimitiveToK(const CNodePtr &primitive_user, size_t index) {
  auto primal = primitive_user->input(index);
  if (!IsValueNode<Primitive>(primal)) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, primitive_user)
      << "Primal graph \"" << primal->ToString() << "\" is not a ValueNode of Primitive.";
  }
  ScopeGuard scope_guard(primal->scope());
  // Map Primitive to K
  auto value_node = primal->cast<ValueNodePtr>();
  auto prim = GetValueNode<PrimitivePtr>(value_node);
  if ((prim->Hash() == prim::kPrimStopGradient->Hash() && prim->name() == prim::kPrimStopGradient->name()) ||
      (prim->Hash() == prim::kPrimUpdateState->Hash() && prim->name() == prim::kPrimUpdateState->name()) ||
      StopGradientForScalar(primitive_user)) {
    MS_LOG(DEBUG) << "Should stop gradient for " << prim->ToString();
    need_cut_ = true;
  }
  if (prim->Hash() == prim::kPrimPyExecute->Hash() && prim->name() == prim::kPrimPyExecute->name()) {
    // Except for the None node
    constexpr size_t script_index = 1;
    auto script_node = primitive_user->input(script_index);
    std::string script = "";
    if (IsValueNode<StringImm>(script_node)) {
      script = GetValueNode<StringImmPtr>(script_node)->value();
    }
    if (script != "None") {
      MS_LOG(WARNING) << "The gradient will be stopped from propagating at the PyExecute node created at the location: "
                      << trace::GetDebugInfoStr(primitive_user->debug_info());
      need_cut_ = true;
    }
  }

  auto k_prim = g_k_prims.KPrimitive(primitive_user, value_node, resources_, is_view_inplace_);
  if (k_prim != nullptr) {
    auto prim_recompute_attr = prim->GetAttr(kAttrRecompute);
    if (prim_recompute_attr != nullptr && prim_recompute_attr->isa<BoolImm>()) {
      auto recomputed = GetValue<bool>(prim_recompute_attr);
      if (recomputed) {
        k_prim->set_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH, true);
      } else {
        k_prim->set_flag(FUNC_GRAPH_NOT_RECOMPUTE_K_GRAPH, true);
      }
    }
    return NewValueNode(k_prim);
  }
  // When failed to find k_prim, try k_meta.
  auto k_meta = g_k_prims.KMetaFuncGraph(prim, primitive_user);
  if (k_meta != nullptr) {
    return NewValueNode(k_meta);
  }
  MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, primitive_user)
    << "Fail to map Primitive of \"" << primal->ToString() << "\" to K.";
}

// Construct representation graph for ValueNode of FuncGraph.
AnfNodePtr DFunctor::MapFuncGraphToK(const AnfNodePtr &primal) {
  if (!IsValueNode<FuncGraph>(primal)) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, primal)
      << "Primal graph \"" << primal->ToString() << "\" is not a ValueNode of FuncGraph.";
  }
  ScopeGuard scope_guard(primal->scope());
  // Map func graph to K
  auto func_graph = GetValueNode<FuncGraphPtr>(primal);
  auto f = func_graph_to_functor_.find(func_graph);
  if (f != func_graph_to_functor_.end()) {
    MS_LOG(DEBUG) << "K graph functor already exist " << func_graph->ToString() << ".";
    return NewValueNode(f->second->k_graph_);
  }
  auto k_user_defined = KUserDefined(func_graph);
  if (k_user_defined != nullptr) {
    MS_LOG(DEBUG) << "K graph functor user defined bprop " << func_graph->ToString() << ".";
    (void)func_graph->transforms().emplace("grad", FuncGraphTransform(k_user_defined));
    (void)k_user_defined->transforms().emplace("custom_bprop_primal", FuncGraphTransform(func_graph));
    return NewValueNode(k_user_defined);
  }
  auto functor = std::make_shared<DFunctor>(func_graph, resources_, false, is_view_inplace_);
  functor->Init();
  functor->MapObject();
  functor->MapMorphism();

  if (func_graph->has_flag(FUNC_GRAPH_FLAG_NO_INLINE)) {
    functor->k_graph_->set_flag(FUNC_GRAPH_FLAG_NO_INLINE, true);
  }
  if (func_graph->has_flag(FUNC_GRAPH_FLAG_CELL_REUSE)) {
    functor->k_graph_->set_flag(FUNC_GRAPH_FLAG_CELL_REUSE, true);
  }
  if (func_graph->has_flag(GRAPH_FLAG_IS_WHILE_HEADER)) {
    functor->k_graph_->set_flag(GRAPH_FLAG_IS_WHILE_HEADER, true);
    functor->tape_->set_flag(GRAPH_FLAG_IS_WHILE_HEADER, true);
  }
  if (func_graph->has_flag(FUNC_GRAPH_OUTPUT_NO_RECOMPUTE)) {
    functor->k_graph_->set_flag(FUNC_GRAPH_RECOMPUTE_K_GRAPH, true);
  }

  MS_LOG(DEBUG) << "Map \"" << func_graph->ToString() << "\" to \"" << functor->k_graph_->ToString() << "\"";
  return NewValueNode(functor->k_graph_);
}

// Construct for ValueNode of Parameter.
AnfNodePtr DFunctor::MapParameterToK(const AnfNodePtr &primal) {
  if (!primal->isa<Parameter>()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, primal)
      << "Primal graph \"" << primal->ToString() << "\" is not a ValueNode of Parameter.";
  }
  ScopeGuard scope_guard(primal->scope());
  // Map Parameter to K
  TraceGuard trace_guard(MakeTraceInfo<TraceGradFprop>(primal->debug_info()));
  auto ret = k_graph_->add_parameter();
  ret->cast_ptr<Parameter>()->set_name(primal->cast_ptr<Parameter>()->name());
  return ret;
}

void DFunctor::MapFvObject() {
  // Map free variable.
  const auto &free_variables_nodes = primal_graph_->free_variables_nodes();
  for (auto &node : free_variables_nodes) {
    ScopeGuard scope_guard(node->scope());
    MS_LOG(DEBUG) << "The free variable " << node->ToString() << ".";
    // Find fv's K from parent.
    AdjointPtr adjoint = nullptr;
    auto parent_adjoint = FindAdjoint(node);
    if (parent_adjoint != nullptr) {
      adjoint = std::make_shared<Adjoint>(node, parent_adjoint->k(), tape_, is_view_inplace_);
    } else {
      if (is_top_ || node->isa<Parameter>()) {
        // Out of ad scope, add adjoint for free variables.
        adjoint = std::make_shared<Adjoint>(node, node, tape_, is_view_inplace_);
        UpdateAdjoint(adjoint);
      } else {
        MS_LOG(DEBUG) << "Fail to find parent adjoint for nontop fv " << node->ToString() << ".";
        adjoint = std::make_shared<Adjoint>(node, nullptr, tape_, is_view_inplace_);
      }
    }
    if (adjoint == nullptr) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node) << "Failed for free variable " << node->ToString() << ".";
    }
    anfnode_to_adjoin_[node] = adjoint;
  }
}

void DFunctor::MapParamObject() {
  // Map parameter.
  for (auto &p : primal_graph_->parameters()) {
    ScopeGuard scope_guard(p->scope());
    MS_LOG(DEBUG) << "The parameter " << p->ToString() << ".";
    auto adjoint = std::make_shared<Adjoint>(p, MapParameterToK(p), tape_, is_view_inplace_);
    UpdateAdjoint(adjoint);
    anfnode_to_adjoin_[p] = adjoint;
  }
}

void DFunctor::MapValueObject() {
  // Map ValueNode.
  auto manager = resources_->manager();
  if (pynative::GradState::Get().RequiresGrad()) {
    CopyPrimitivePtrForFpropReplace(primal_graph_, manager);
  }
  if (is_view_inplace_) {
    DuplicateSideEffectNodes(manager, primal_graph_);
  }
  auto &value_nodes = primal_graph_->value_nodes();
  for (const auto &value_pair : value_nodes) {
    auto node = value_pair.first;
    auto parent_adjoint = FindAdjoint(node);
    if (parent_adjoint != nullptr) {
      auto adjoint = std::make_shared<Adjoint>(node, parent_adjoint->k(), tape_, is_view_inplace_);
      anfnode_to_adjoin_[node] = adjoint;
      continue;
    }

    AdjointPtr adjoint = nullptr;
    if (IsValueNode<Primitive>(node)) {  // Primitive.
      if (PrimitiveNeedSkip(node)) {
        continue;
      }
      MS_LOG(DEBUG) << "Map Primitive node " << node->DebugString() << ".";
      auto &users = manager->node_users()[node];
      if (users.size() == 0) {
        MS_LOG(ERROR) << "\"" << node->DebugString() << "\" has no user.";
        continue;
      } else if (users.size() > 1) {
        MS_LOG(DEBUG) << "\"" << node->DebugString() << "\" supposed to be used once, but users size: " << users.size();
      }
      auto cnode = users.begin()->first->cast<CNodePtr>();  // We just use the first user.
      auto index = users.begin()->second;
      adjoint = std::make_shared<Adjoint>(node, MapPrimitiveToK(cnode, index), tape_, is_view_inplace_);
    } else if (IsValueNode<FuncGraph>(node)) {  // FuncGraph
      MS_LOG(DEBUG) << "Map FuncGraph node " << node->DebugString() << ".";
      adjoint = std::make_shared<Adjoint>(node, MapFuncGraphToK(node), tape_, is_view_inplace_);
    } else if (node->isa<Parameter>()) {  // Parameter, hardly reach here.
      MS_LOG(DEBUG) << "Map Parameter node " << node->DebugString() << ".";
      adjoint = std::make_shared<Adjoint>(node, MapParameterToK(node), tape_, is_view_inplace_);
    } else {
      adjoint = std::make_shared<Adjoint>(node, node, tape_, is_view_inplace_);
    }
    UpdateAdjoint(adjoint);
    anfnode_to_adjoin_[node] = adjoint;
  }
}

// Skip morphism.
void DFunctor::MapObject() {
  // The order does not matter
  MapFvObject();
  MapParamObject();
  MapValueObject();
}

void DFunctor::UpdateAdjoint(const AdjointPtr &adjoint_definition) {
  auto primal = adjoint_definition->primal();
  if (anfnode_to_adjoin_definition_.find(primal) != anfnode_to_adjoin_definition_.end()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, primal) << "UpdateAdjoint adjoint definition already exists "
                                                 << primal_graph_->ToString() << " " << primal->ToString() << ".";
  }
  anfnode_to_adjoin_definition_[primal] = adjoint_definition;
  // Update k hole for primal.
  for (auto &f : func_graph_to_functor_) {
    auto adjoint = f.second->anfnode_to_adjoin_.find(primal);
    if (adjoint != f.second->anfnode_to_adjoin_.end()) {
      adjoint->second->UpdateK(adjoint_definition->k());
    }
    adjoint = f.second->anfnode_to_adjoin_indirect_fv_.find(primal);
    if (adjoint != f.second->anfnode_to_adjoin_indirect_fv_.end()) {
      adjoint->second->UpdateK(adjoint_definition->k());
    }
  }
}

AdjointPtr DFunctor::FindAdjoint(const AnfNodePtr &primal) const {
  auto adjoint = anfnode_to_adjoin_definition_.find(primal);
  if (adjoint != anfnode_to_adjoin_definition_.end()) {
    MS_LOG(DEBUG) << "Found adjoint definition for free variable " << primal->ToString() << ".";
    return adjoint->second;
  }
  MS_LOG(DEBUG) << "The adjoint definition for free variable not defined yet " << primal->ToString() << ".";
  return nullptr;
}

void DFunctor::CallDoutHoleOnTape() const {
  if (!is_top_) {
    return;
  }

  // Call dout hole of all adjoint.
  for (auto &f : func_graph_to_functor_) {
    for (auto &adjoint : f.second->anfnode_to_adjoin_) {
      adjoint.second->CallDoutHole();
    }
    for (auto &adjoint : f.second->anfnode_to_adjoin_indirect_fv_) {
      adjoint.second->CallDoutHole();
    }
  }
}

FuncGraphPtr DFunctor::k_graph() { return k_graph_; }

FuncGraphPtr DFunctor::tape() { return tape_; }

void DFunctor::BroadCastStopFlag() {
  // As stop set expanding, all directly or indirectly stopped CNode will be cut off
  const auto &node_users_map = resources_->manager()->node_users();
  while (need_cut_) {
    need_cut_ = false;
    for (auto &node : primal_graph_->nodes()) {
      auto cnode = dyn_cast<CNode>(node);
      if (cnode != nullptr && !cnode->stop_gradient()) {
        if (NeedGradForUpdateState(cnode, node_users_map, is_view_inplace_)) {
          continue;
        }
        // Cut off the cnode only when it's not referred any more
        if (cnode->IsApply(prim::kPrimStopGradient) || cnode->IsApply(prim::kPrimUpdateState) ||
            AllReferencesStopped(cnode) || StopGradientForScalar(cnode) || cnode->IsApply(prim::kPrimPyExecute)) {
          MS_LOG(DEBUG) << "Set stop gradient flag for " << cnode->ToString() << ".";
          cnode->set_stop_gradient(true);
          // The stop set changed, more cut required
          need_cut_ = true;
        }
      }
    }
  }
}

bool DFunctor::AllReferencesStopped(const CNodePtr &node) {
  auto &users = primal_graph_->manager()->node_users()[node];
  // Only care about stop_gradient caused cutting
  if (users.empty()) {
    return false;
  }
  for (auto &kv : users) {
    auto &user = kv.first;
    if (!user->isa<CNode>()) {
      return false;
    } else {
      auto cnode = user->cast_ptr<CNode>();
      MS_EXCEPTION_IF_NULL(cnode);
      if (!cnode->stop_gradient()) {
        return false;
      }
    }
  }
  return true;
}

CNodePtr GetJUser(const NodeUsersMap &node_user_map, const CNodePtr &cnode, int index) {
  constexpr auto recursive_level = 2;
  auto it = node_user_map.find(cnode);
  if (it == node_user_map.end()) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode)
      << "J CNode not used {" << cnode->DebugString(recursive_level) << "/" << index << "}";
  }
  auto &j_users = it->second;
  auto size = j_users.size();
  if (size != 1) {
    bool has_multiple_j_call_user = false;
    CNodePtr j_call_user = nullptr;
    for (auto &user : j_users) {
      // If J CNode is used as a FV, the j_users.size may exceed 1 user. It is allowed.
      if (user.second == 0) {
        // Real J CNode call user.
        if (j_call_user == nullptr) {  // First user.
          j_call_user = user.first->cast<CNodePtr>();
        } else {  // More than 1 call user. Not allowed.
          has_multiple_j_call_user = true;
        }
      }
    }
    if (has_multiple_j_call_user) {  // Has multiple J CNode call user.
      std::ostringstream user_info;
      for (auto &user : j_users) {
        user_info << "    user: " << user.first->DebugString() << ", index: " << user.second << "\n";
      }
#ifdef ENABLE_DUMP_IR
      DumpIR("J_User_Ex_" + cnode->func_graph()->ToString() + ".ir", cnode->func_graph());
#endif
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cnode)
        << "Incorrect J CNode user size: " << size << ", of {" << cnode->DebugString(recursive_level) << "/" << index
        << "}\nUser Info:\n"
        << user_info.str();
    } else {
      return j_call_user;
    }
  }
  return j_users.begin()->first->cast<CNodePtr>();
}

CNodePtr GetPrimalUser(const CNodePtr &j_user, const std::map<FuncGraphPtr, std::vector<CNodePtr>> &primal_map) {
  // Check if the forward network and the gradient of it are called in the same graph.
  auto graph = j_user->func_graph();
  auto iter = primal_map.find(graph);
  if (iter == primal_map.end()) {
    // The CNode using the forward graph result and the gradient of the forward graph are not in the same graph.
    // The EliminatePrimalGraph optimization can not be done. If the code use the forward network and its gradient,
    // the forward network can not be eliminated. This may cause the decrease of the compilation and running efficiency.
    MS_LOG(DEBUG) << "The gradient operation of forward network and the forward network are not called in the same"
                  << " graph. The CNode to use the gradient result is: " << j_user->DebugString()
                  << " This CNode is in graph: " << graph->ToString();
    return nullptr;
  }

  // Check if there is only one primal call corresponding to the specified j user.
  auto primal_users = iter->second;
  if (primal_users.size() != 1) {
    MS_LOG(WARNING) << "It is recommended to call the forward network only once.";
    MS_LOG(INFO) << "There is " << primal_users.size()
                 << " primal calls for same J operation in the same graph. Func graph: " << graph->ToString()
                 << ", J operation: " << j_user->DebugString() << ", Primal call: ";
    size_t count = 0;
    for (const auto &user : primal_users) {
      MS_LOG(INFO) << "[ " << ++count << " ] : " << user->DebugString(2) << trace::DumpSourceLines(user, false);
    }
    return nullptr;
  }

  // Check input size.
  auto primal_user = primal_users[0];
  if (primal_user->size() != j_user->size()) {
    MS_LOG(WARNING) << "Input size incorrect, the input size of primal " << primal_user->DebugString() << " is "
                    << primal_user->size() << ", and J user " << j_user->DebugString() << " is " << j_user->size();
    return nullptr;
  }
  return primal_user;
}

static mindspore::HashMap<CNodePtr, std::vector<CNodePtr>> FindPrimalJPair(const FuncGraphManagerPtr &manager,
                                                                           const FuncGraphPtr &primal_graph) {
  std::vector<CNodePtr> j_users;
  std::map<FuncGraphPtr, std::vector<CNodePtr>> primal_map;
  const auto &node_user_map = manager->node_users();
  // Search primal graph user cnodes.
  for (auto &entry : primal_graph->func_graph_cnodes_index()) {
    auto cnode = entry.first->first->cast<CNodePtr>();
    auto index = entry.first->second;
    if (index == 0) {
      // To find real calling.
      auto fg = cnode->func_graph();
      MS_EXCEPTION_IF_NULL(fg);
      const auto &iter = primal_map.find(fg);
      if (iter != primal_map.end()) {
        iter->second.push_back(cnode);
        continue;
      }
      primal_map[fg] = {cnode};
    } else if (IsPrimitive(cnode->inputs().at(0), prim::kPrimJ)) {
      // To find J user.
      j_users.emplace_back(GetJUser(node_user_map, cnode, index));
    }
  }

  mindspore::HashMap<CNodePtr, std::vector<CNodePtr>> primal_user_to_j_users;
  for (const auto &j_user : j_users) {
    MS_EXCEPTION_IF_NULL(j_user);
    auto primal = GetPrimalUser(j_user, primal_map);
    if (primal == nullptr) {
      continue;
    }
    MS_LOG(DEBUG) << "Primal_J pair is found, where primal is: " << primal->DebugString()
                  << " and J user is: " << j_user->DebugString();
    primal_user_to_j_users[primal].emplace_back(j_user);
  }
  return primal_user_to_j_users;
}

static void RemovePrimalUpdateStates(const FuncGraphManagerPtr &manager, const CNodePtr &primal_call) {
  auto &node_users = manager->node_users();
  auto iter = node_users.find(primal_call);
  if (iter == node_users.end()) {
    // Skip if user of primal_call not found.
    return;
  }
  // Find UpdateState nodes after the primal call.
  std::vector<CNodePtr> update_states;
  for (auto &user : iter->second) {
    auto &user_node = user.first;
    if (IsPrimitiveCNode(user_node, prim::kPrimUpdateState)) {
      update_states.emplace_back(user_node->cast<CNodePtr>());
    }
  }
  // Remove UpdateStates by replace them with their monad input.
  for (auto &update_state : update_states) {
    auto &input_monad = update_state->inputs().at(1);
    (void)manager->Replace(update_state, input_monad);
  }
}

static bool CopyMonadArguments(const CNodePtr &primal_user, const CNodePtr &j_user,
                               const FuncGraphManagerPtr &manager) {
  auto &primal_inputs = primal_user->inputs();
  auto &j_user_inputs = j_user->inputs();
  bool has_monad = false;
  for (size_t i = 1; i < primal_inputs.size(); ++i) {
    auto &input = primal_inputs.at(i);
    if (HasAbstractMonad(input)) {
      // Copy monad input from primal to j_user.
      manager->SetEdge(j_user, i, input);
      has_monad = true;
    } else if (input != j_user_inputs.at(i)) {
      // Skip if there are different non-monad inputs.
      return false;
    }
  }
  return has_monad;
}

//
// To replace the primal graph with k graph.
// Convert:
//   x = primal(args, u0)
//   u1 = update_state(u0, x)
//   ...
//   tuple = K(args, u1)
//   u2 = update_state(u1, tuple)
//   ...
// To:
//   tuple = K(args, u0)
//   x = get_item(tuple, 0)
//   ...
//   tuple = K(args, u0)
//   u2 = update_state(u0, tuple)
//   ...
//
void DFunctor::EliminatePrimalGraph() {
  // Find primal user and paired J user cnodes.
  auto manager = primal_graph_->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto primal_user_to_j_users = FindPrimalJPair(manager, primal_graph_);
  for (const auto &iter : primal_user_to_j_users) {
    auto primal_user = iter.first;
    auto &j_users = iter.second;
    MS_EXCEPTION_IF_NULL(primal_user);
    if (j_users.size() == 1) {
      // If both inputs are same except monads, we copy primal monad args to k graph
      // so that they can be combined in CSE (common subexpression elimination) pass.
      // Only do this when the size of j_users is 1 in order to keep the execution order.
      const bool has_monad = CopyMonadArguments(primal_user, j_users[0], manager);
      // Remove the UpdateState nodes after primal_user if need.
      if (has_monad) {
        RemovePrimalUpdateStates(manager, primal_user);
      }
    } else {
      MS_LOG(INFO) << "There are multiple j users with the same primal user " << primal_user->DebugString();
    }

    // Replace primal graph with k graph.
    auto k_vnode = NewValueNode(k_graph_);
    primal_user->set_input(0, k_vnode);
    if (j_users.empty()) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, primal_user) << "The J nodes for primal graph " << primal_graph_->ToString()
                                                        << " should be used by at least one other node.";
    }
    primal_user->set_abstract(j_users[0]->abstract());
    // Insert tuple_getitem after primal user cnode.
    auto construct_wrapper = primal_user->func_graph();
    auto tuple_getitem = NewValueNode(prim::kPrimTupleGetItem);
    auto imm0 = std::make_shared<Int64Imm>(0);
    auto idx0 = NewValueNode(SizeToLong(0));
    idx0->set_abstract(std::make_shared<abstract::AbstractScalar>(imm0));
    auto getitem0 = construct_wrapper->NewCNode({tuple_getitem, primal_user, idx0});
    getitem0->CloneCNodeInfo(primal_user);
    (void)manager->Replace(primal_user, getitem0);
  }
}
}  // namespace ad
}  // namespace mindspore
