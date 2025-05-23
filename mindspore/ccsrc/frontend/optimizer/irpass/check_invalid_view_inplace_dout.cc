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
#include "frontend/optimizer/irpass/check_invalid_view_inplace_dout.h"

#include <algorithm>
#include <memory>
#include "include/common/utils/utils.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/optimizer.h"
#include "include/common/pynative/variable.h"
#include "mindspore/core/include/base/base_ref.h"
#include "mindspore/core/include/ir/func_graph_cloner.h"
#include "mindspore/core/include/ir/anf.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ccsrc/pipeline/jit/ps/pass.h"

namespace mindspore {
namespace opt {
namespace irpass {

namespace {
bool IsCreatedByViewOp(const AnfNodePtr &node) {
  const auto &prim = GetCNodePrimitive(node);
  if (prim != nullptr && prim->graph_view_prim()) {
    return true;
  }
  auto abstract = node->abstract();
  if (abstract != nullptr && abstract->isa<abstract::AbstractRefTensor>()) {
    if (abstract->cast_ptr<abstract::AbstractRefTensor>()->is_view()) {
      return true;
    }
  }
  return false;
}

std::vector<std::size_t> NeedCheckInplaceCNode(const CNodePtr &primal_cnode, const FuncGraphPtr &bprop_fg) {
  MS_EXCEPTION_IF_NULL(primal_cnode);
  const auto &prim = GetCNodePrimitive(primal_cnode);
  // InplaceOp(inplace_changed_input0, ... , unchanged_input0, ...)
  if (prim == nullptr || !prim->inplace_prim()) {
    return {};
  }
  const auto &inputs = primal_cnode->inputs();
  const auto &rw_write_index = prim->rw_write_input_indexes();
  bool exist_view_inplace = std::any_of(rw_write_index.begin(), rw_write_index.end(), [&inputs](const size_t &index) {
    return IsCreatedByViewOp(inputs[index + 1]);
  });
  if (!exist_view_inplace) {
    return {};
  }
  std::vector<std::size_t> result;
  for (size_t input_index = 1; input_index < inputs.size(); ++input_index) {
    const auto &input_abs = inputs[input_index]->abstract();
    if (input_abs != nullptr &&
        (input_abs->isa<abstract::AbstractScalar>() || input_abs->isa<abstract::AbstractMonad>())) {
      continue;
    }
    bool is_need_check_index =
      std::any_of(rw_write_index.begin(), rw_write_index.end(),
                  [&input_index](const size_t &rw_index) { return rw_index == input_index - 1; });
    if (!is_need_check_index) {
      (void)result.emplace_back(input_index);
      MS_LOG(DEBUG) << "Mark inplace primitive's " << input_index << " input as invalid differentiation target.";
    }
  }
  bprop_fg->set_attr(kInvalidInplaceDout, MakeValue(result));
  return result;
}

void ResetUselessFuncGraph(const FuncGraphPtr &func_graph, const std::vector<bool> &use_flag) {
#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("opt_check_invalid_dout_before_reset_useless_index_" + func_graph->ToString() + ".ir", func_graph);
  }
#endif
  auto bprop_output = func_graph->output();
  MS_LOG(INFO) << "Reset useless element for bprop_output: " << bprop_output->DebugString();
  if (!IsPrimitiveCNode(bprop_output, prim::kPrimMakeTuple)) {
    return;
  }
  auto bprop_cnode = bprop_output->cast<CNodePtr>();
  auto cnode_inputs = bprop_cnode->inputs();
  for (size_t i = 1; i < cnode_inputs.size(); ++i) {
    if (!use_flag[i - 1]) {
      bprop_cnode->set_input(i, NewValueNode(MakeValue<int64_t>(0)));
      MS_LOG(INFO) << "Set bprop cnode's input: " << i << " , as constant zero.";
    }
  }
#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    DumpIR("opt_check_invalid_dout_after_reset_useless_index_" + func_graph->ToString() + ".ir", func_graph);
  }
#endif
}

bool CheckInvalidDoutOfBprop(const FuncGraphPtr &func_graph, bool need_throw_exception = true) {
  const auto &all_nodes = TopoSort(func_graph->output(), SuccDeeperSimple);
  for (auto &node : all_nodes) {
    auto sub_fg = GetValueNode<FuncGraphPtr>(node);
    if (sub_fg == nullptr || !sub_fg->has_flag(kFlagNeedCheckViewInplaceDoutBprop)) {
      continue;
    }
    const auto &indexes = GetValue<std::vector<std::size_t>>(sub_fg->get_attr(kInvalidInplaceDout));
    auto bprop_output = sub_fg->output();
    MS_EXCEPTION_IF_NULL(bprop_output);
    auto abstract = bprop_output->abstract();
    MS_EXCEPTION_IF_NULL(abstract);
    MS_LOG(DEBUG) << "Bprop sub func_graph output: " << bprop_output->DebugString()
                  << " , abstract: " << abstract->ToString();
    auto seq_node = (*abstract->cast<abstract::AbstractSequencePtr>()->sequence_nodes())[0];
    auto bprop_output_indexes = GetSequenceNodeElementsUseFlags(seq_node.lock());
    if (bprop_output_indexes == nullptr) {
      continue;
    }
    bool has_invalid_dout = std::any_of(indexes.begin(), indexes.end(), [&bprop_output_indexes](const auto index) {
      return (*bprop_output_indexes)[index];
    });
    if (has_invalid_dout) {
      if (need_throw_exception) {
        MS_LOG(EXCEPTION)
          << "When performing an in-place operation on an object generated by a view operation, it "
             "is currently not supported to compute gradients for the other inputs of this in-place operator.";
      }
      return true;
    }
    sub_fg->erase_flag(kFlagNeedCheckViewInplaceDoutBprop);
  }
  return false;
}

// For AbstractSequence, if elements_use_flag is not all true, reset it
// All elements_use_flag are true means this tuple may be used as total
void SetAbstractSeqElementsUseFlags(const AbstractBasePtr &abstract) {
  if (abstract == nullptr || !abstract->isa<abstract::AbstractSequence>()) {
    return;
  }
  auto seq_abstract = abstract->cast<abstract::AbstractSequencePtr>();
  MS_EXCEPTION_IF_NULL(seq_abstract);
  auto seq_nodes = seq_abstract->sequence_nodes();
  if (seq_nodes == nullptr) {
    return;
  }
  for (auto seq_node : *seq_nodes) {
    auto indexes = GetSequenceNodeElementsUseFlags(seq_node.lock());
    bool no_need_reset = std::all_of((*indexes).begin(), (*indexes).end(), [](const auto index) { return index; });
    if (!no_need_reset) {
      SetSequenceElementsUseFlagsRecursively(seq_abstract, false);
      return;
    }
  }
}

AbstractBasePtr GetFuncGraphOutputAbstract(const FuncGraphPtr &func_graph) {
  if (func_graph != nullptr && func_graph->output() != nullptr) {
    return func_graph->output()->abstract();
  }
  return nullptr;
}

FuncGraphPtr GetAbstractFuncGraph(const abstract::AbstractFuncAtomPtr &abs) {
  if (abs->isa<abstract::FuncGraphAbstractClosure>()) {
    auto abstract_func_graph = abs->cast<abstract::FuncGraphAbstractClosurePtr>();
    return abstract_func_graph->func_graph();
  }
  if (abs->isa<abstract::PartialAbstractClosure>()) {
    auto abstract_partial_func = abs->cast<abstract::PartialAbstractClosurePtr>();
    auto abstract_fn = abstract_partial_func->fn();
    if (abstract_fn != nullptr && abstract_fn->isa<abstract::FuncGraphAbstractClosure>()) {
      auto abstract_func_graph = abstract_fn->cast<abstract::FuncGraphAbstractClosurePtr>();
      return abstract_func_graph->func_graph();
    }
  }
  return nullptr;
}

void ReMarkElementsUseFlag(const FuncGraphPtr &func_graph) {
  auto func_graph_output = func_graph->output();
  const auto &all_nodes = TopoSort(func_graph_output, SuccDeeperSimple);
  // Set AbstractSequence's element use flag as false
  for (auto &node : all_nodes) {
    auto cnode = node->cast<CNodePtr>();
    if (node == func_graph_output || cnode == nullptr || cnode->abstract() == nullptr ||
        !cnode->abstract()->isa<abstract::AbstractSequence>()) {
      continue;
    }
    // For maketuple and call node, just change cnode's abstract
    if (IsPrimitiveCNode(cnode, prim::kPrimMakeTuple) || GetValueNode<FuncGraphPtr>(cnode->input(0)) != nullptr) {
      SetAbstractSeqElementsUseFlags(cnode->abstract());
    } else if (auto abstract = cnode->input(0)->abstract();
               abstract != nullptr && abstract->isa<abstract::AbstractFuncUnion>()) {
      auto func_list = dyn_cast<abstract::AbstractFuncUnion>(abstract)->func_list();
      for (auto func : func_list) {
        auto fg = GetAbstractFuncGraph(func);
        SetAbstractSeqElementsUseFlags(GetFuncGraphOutputAbstract(fg));
      }
    }
  }

  // Remark AbstractSequence's element use flag according to tuplegetitem
  for (auto node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto tuple = cnode->input(kIndex1);
    auto index_value = GetValueNode<Int64ImmPtr>(cnode->input(kIndex2));
    size_t index = LongToSize(index_value->value());
    SetSequenceElementsUseFlags(tuple->abstract(), index, true);
    auto tuple_cnode = tuple->cast<CNodePtr>();
    if (tuple_cnode == nullptr) {
      continue;
    }
    auto abstract = tuple_cnode->input(0)->abstract();
    if (abstract == nullptr || !abstract->isa<abstract::AbstractFuncUnion>()) {
      continue;
    }
    auto func_list = dyn_cast<abstract::AbstractFuncUnion>(abstract)->func_list();
    for (auto func : func_list) {
      auto fg = GetAbstractFuncGraph(func);
      auto output_abstract = GetFuncGraphOutputAbstract(fg);
      if (output_abstract != nullptr) {
        SetSequenceElementsUseFlags(output_abstract, index, true);
      }
    }
  }
}

bool EraseUnUsedNode(const FuncGraphPtr &func_graph) {
  bool is_changed = false;
  const auto &all_nodes = TopoSort(func_graph->output(), SuccDeeperSimple);
  for (auto &node : all_nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimMakeTuple)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    auto abstract = cnode->abstract()->cast<abstract::AbstractSequencePtr>();
    MS_EXCEPTION_IF_NULL(abstract);
    auto seq_nodes = abstract->sequence_nodes();
    if (seq_nodes == nullptr) {
      continue;
    }
    for (auto seq_node : *seq_nodes) {
      const auto &indexes = GetSequenceNodeElementsUseFlags(seq_node.lock());
      for (size_t i = 0; i < (*indexes).size(); ++i) {
        // Set unused node as I64(0)
        if (!(*indexes)[i] && GetValueNode<Int64ImmPtr>(cnode->input(i + 1)) == nullptr) {
          cnode->set_input(i + 1, NewValueNode(MakeValue<int64_t>(0)));
          is_changed = true;
        }
      }
    }
  }
  return is_changed;
}

void CheckInvalidDoutFromElementsUseFlag(const FuncGraphPtr &func_graph, bool need_clone = false) {
  // Check invalid dout for the first time
  // If passed, just return
  // If has invalid dout, do second check to avoid incorrect element_use_flag

#ifdef ENABLE_DUMP_IR
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  if (context->CanDump(kIntroductory)) {
    DumpIR("opt_check_invalid_dout_before_first_check" + func_graph->ToString() + ".ir", func_graph);
  }
#endif

  if (!CheckInvalidDoutOfBprop(func_graph, false)) {
    return;
  }

  FuncGraphPtr check_func_graph = func_graph;
  if (need_clone) {
    check_func_graph = BasicClone(func_graph);
  }

  // Remark element use flag and remove unused nodes until not change
  bool is_changed = true;
  while (is_changed) {
    ReMarkElementsUseFlag(check_func_graph);
    is_changed = EraseUnUsedNode(check_func_graph);
  }

#ifdef ENABLE_DUMP_IR
  if (context->CanDump(kIntroductory)) {
    DumpIR("opt_check_invalid_dout_before_second_check" + check_func_graph->ToString() + ".ir", check_func_graph);
  }
#endif
  CheckInvalidDoutOfBprop(check_func_graph);
}
}  // namespace

// For GraphMode and grad under @jit
bool CheckInvalidViewInplaceDout::operator()(const FuncGraphPtr &root, const OptimizerPtr &opt) {
  MS_EXCEPTION_IF_NULL(root);
  CheckInvalidDoutFromElementsUseFlag(root, true);
  return false;
}

// For Gradjit situation
void CheckBpropGraphHasInvalidDoutHelper(const FuncGraphPtr &func_graph, const std::vector<bool> &need_grads) {
  MS_EXCEPTION_IF_NULL(func_graph);
  bool need_check = false;
  for (const auto &node : TopoSort(func_graph->output(), SuccDeeperSimple)) {
    auto sub_fg = GetValueNode<FuncGraphPtr>(node);
    if (sub_fg != nullptr && sub_fg->has_flag(kFlagNeedCheckViewInplaceDoutBprop)) {
      need_check = true;
      break;
    }
  }
  if (!need_check) {
    return;
  }
  FuncGraphPtr cloned_func_graph = BasicClone(func_graph);
  ResetUselessFuncGraph(cloned_func_graph, need_grads);
  auto resource = std::make_shared<pipeline::Resource>();
  resource->set_func_graph(cloned_func_graph);
  auto new_manager = resource->manager();
  MS_EXCEPTION_IF_NULL(new_manager);
  new_manager->AddFuncGraph(cloned_func_graph);
  auto new_func_graph = CheckInvalidDoutGraphPass(resource);
  CheckInvalidDoutFromElementsUseFlag(new_func_graph);
}

void MarkInvalidInplaceOpDout(const FuncGraphPtr &fprop_graph) {
  bool need_check_dout = false;
  for (const auto &node : TopoSort(fprop_graph->return_node(), SuccDeeperSimple)) {
    // Check fprop graph for each prim
    auto k_fg = GetValueNode<FuncGraphPtr>(node);
    if (k_fg == nullptr) {
      continue;
    }
    // Find primal cnode for this fprop
    const auto &primal_cnode_iter = k_fg->transforms().find("primal_cnode");
    if (primal_cnode_iter == k_fg->transforms().end()) {
      continue;
    }
    const auto &primal_cnode = primal_cnode_iter->second.primal_cnode();
    MS_EXCEPTION_IF_NULL(k_fg->output());
    auto k_fg_output_cnode = k_fg->output()->cast<CNodePtr>();
    auto bprop_node = k_fg_output_cnode->input(2);
    auto bprop_fg = GetValueNode<FuncGraphPtr>(bprop_node);
    const auto &check_inputs = NeedCheckInplaceCNode(primal_cnode, bprop_fg);
    if (!check_inputs.empty()) {
      bprop_fg->set_flag(kFlagNeedCheckViewInplaceDoutBprop, true);
      bprop_fg->set_flag(FUNC_GRAPH_FLAG_DEFER_INLINE, true);
      need_check_dout = true;
    }
  }
  if (need_check_dout) {
    MS_LOG(INFO) << "Mark invalid inplace dout for func_graph: " << fprop_graph->ToString();
#ifdef ENABLE_DUMP_IR
    auto context = MsContext::GetInstance();
    MS_EXCEPTION_IF_NULL(context);
    if (context->CanDump(kIntroductory)) {
      DumpIR("opt_check_invalid_dout_mark_invalid_dout_" + fprop_graph->ToString() + ".ir", fprop_graph);
    }
#endif
  }
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
