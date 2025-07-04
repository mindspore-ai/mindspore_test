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

#include "frontend/optimizer/irpass/branch_culling.h"

#include <memory>
#include <utility>
#include <queue>

#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "utils/hash_map.h"
#include "ir/func_graph.h"
#include "ir/func_graph_cloner.h"
#include "frontend/operator/ops.h"
#include "frontend/optimizer/irpass.h"
#include "frontend/optimizer/pattern_matcher.h"
#include "pipeline/jit/ps/parse/resolve.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_e.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_h.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_o.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
namespace irpass {
constexpr size_t kCondIndex = 1;
constexpr size_t kTrueBranchIndex = 2;
constexpr size_t kFalseBranchIndex = 3;
namespace internal {
AnfNodePtr GenerateSwitchNode(const FuncGraphPtr &graph, const AnfNodePtr &cond, const AnfNodePtr &data,
                              int64_t switch_idx) {
  auto switch_node = prim::GetPythonOps("geswitch", "mindspore.ops.functional")->cast<PrimitivePtr>();
  std::vector<AnfNodePtr> switch_nodes{NewValueNode(switch_node), data, cond};
  auto switch_apply = graph->NewCNode(switch_nodes);
  std::vector<AnfNodePtr> tuple_getitem_nodes{NewValueNode(prim::kPrimTupleGetItem), switch_apply,
                                              NewValueNode(MakeValue(switch_idx))};
  return graph->NewCNode(tuple_getitem_nodes);
}

AnfNodePtr GenerateSwitchTrueNode(const FuncGraphPtr &graph, const AnfNodePtr &cond, const AnfNodePtr &data) {
  return GenerateSwitchNode(graph, cond, data, 1);
}

AnfNodePtr GenerateSwitchFalseNode(const FuncGraphPtr &graph, const AnfNodePtr &cond, const AnfNodePtr &data) {
  return GenerateSwitchNode(graph, cond, data, 0);
}

bool InConvertWhiteList(const AnfNodePtr &node, size_t index) {
  // The CNode inputs of the following Primitive with index in std::vector<size_t> should not be guarded by geswitch
  // node because it is attribute or ge specific reason.
  // Example : when convert CNode(kPrimReduceSum, x, axis), node of index 2 in CNode->inputs is axis which should not be
  // converted to switch guarded.
  std::vector<std::pair<PrimitivePtr, std::vector<size_t>>> white_list({{prim::kPrimApplyMomentum, {1, 2}},
                                                                        {prim::kPrimMomentum, {2, 3}},
                                                                        {prim::kPrimStateSetItem, {1}},
                                                                        {prim::kPrimTupleGetItem, {2}},
                                                                        {prim::kPrimEnvironGet, {1}},
                                                                        {prim::kPrimEnvironSet, {1}},
                                                                        {prim::kPrimReduceSum, {2}},
                                                                        {prim::kPrimReduceMean, {2}},
                                                                        {prim::kPrimReduceAll, {2}},
                                                                        {prim::kPrimCast, {2}},
                                                                        {prim::kPrimTranspose, {2}},
                                                                        {prim::kPrimOneHot, {2}},
                                                                        {prim::kPrimGather, {3}},
                                                                        {prim::kPrimReshape, {2}},
                                                                        {prim::kPrimAssign, {1}},
                                                                        {prim::kPrimAssignAdd, {1}},
                                                                        {prim::kPrimAssignSub, {1}},
                                                                        {prim::kPrimTensorSummary, {1}},
                                                                        {prim::kPrimImageSummary, {1}},
                                                                        {prim::kPrimScalarSummary, {1}},
                                                                        {prim::kPrimApplyRMSProp, {6, 7, 8}},
                                                                        {prim::kPrimCumSum, {2}},
                                                                        {prim::kPrimTile, {2}},
                                                                        {prim::kPrimExpandDims, {2}},
                                                                        {prim::kPrimHistogramSummary, {1}}});
  for (auto &item : white_list) {
    auto matched = std::any_of(item.second.begin(), item.second.end(), [&item, &node, &index](size_t idx) {
      return IsPrimitiveCNode(node, item.first) && idx == index;
    });
    if (matched) {
      return true;
    }
  }

  std::vector<PrimitivePtr> adapter_convert_ops = {prim::kPrimDepend, prim::kPrimLoad};
  for (auto &item : adapter_convert_ops) {
    if (IsPrimitiveCNode(node, item)) {
      return true;
    }
  }
  return false;
}

using NodeInputReplMap = mindspore::HashMap<std::pair<AnfNodePtr, size_t>, AnfNodePtr, PairHasher>;
// replace the nodes which should be changed
void RunSwitchNodeReplace(const FuncGraphManagerPtr &manager, std::vector<std::pair<CNodePtr, CNodePtr>> nodes_changed,
                          mindspore::HashMap<AnfNodePtr, AnfNodePtr> repl_node, NodeInputReplMap repl_node_inputs,
                          const FuncGraphPtr &func_graph) {
  for (auto &node_pair : nodes_changed) {
    CNodePtr old_node = node_pair.first;
    CNodePtr new_node = node_pair.second;
    MS_EXCEPTION_IF_NULL(old_node);
    MS_EXCEPTION_IF_NULL(new_node);
    for (size_t i = 0; i < old_node->size(); i++) {
      auto input = old_node->input(i);
      if (repl_node.count(input) != 0) {
        new_node->add_input(repl_node[input]);
      } else if (repl_node_inputs.count(std::pair<AnfNodePtr, size_t>(old_node, i)) != 0) {
        new_node->add_input(repl_node_inputs[std::pair<AnfNodePtr, size_t>(old_node, i)]);
      } else {
        new_node->add_input(input);
      }
    }
  }

  for (auto &item : repl_node) {
    if (IsPrimitiveCNode(item.second, prim::kPrimReturn)) {
      func_graph->set_output(item.second->cast<CNodePtr>()->input(1));
    } else if (!manager->Replace(item.first, item.second)) {
      constexpr auto kDebugStrDepth = 2;
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, item.first)
        << "TransformGraphDependNode replace node failed original:" << item.first->DebugString(kDebugStrDepth)
        << " to new: " << item.second->DebugString(kDebugStrDepth);
    }
  }
}

bool HasDependencyOnSubGraph(const FuncGraphPtr &graph, const AnfNodePtr &state) {
  std::queue<AnfNodePtr> nodes;
  nodes.push(state);
  while (!nodes.empty()) {
    auto cur_node = nodes.front();
    MS_EXCEPTION_IF_NULL(cur_node);
    nodes.pop();
    if (cur_node->isa<Parameter>() || cur_node->isa<ValueNode>()) {
      continue;
    }
    if (cur_node->func_graph() == graph) {
      return true;
    }
    auto cur_cnode = cur_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cur_cnode);
    for (size_t i = 1; i < cur_cnode->size(); i++) {
      nodes.push(cur_cnode->input(i));
    }
  }
  return false;
}

// trace the node that should add switch and replace them with new nodes in the graph
FuncGraphPtr TransformGraphCondBranchNodes(
  const FuncGraphPtr &graph, const AnfNodePtr &cond,
  const std::function<AnfNodePtr(FuncGraphPtr graph, AnfNodePtr cond, AnfNodePtr data)> &generate_func) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  // record the node that has been changed
  std::vector<std::pair<CNodePtr, CNodePtr>> nodes_changed;
  // record the node to be replaced
  mindspore::HashMap<AnfNodePtr, AnfNodePtr> repl_node;
  // record the node input to be replaced
  NodeInputReplMap repl_node_inputs;
  const AnfNodeSet &nodes = graph->nodes();
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    auto inputs = node->cast<CNodePtr>()->inputs();
    bool should_replace = false;
    // if the apply input does not belong to graph, insert a switch node
    for (size_t index = 0; index < inputs.size(); index++) {
      auto input_node = inputs[index];
      MS_EXCEPTION_IF_NULL(input_node);
      if (HasAbstractMonad(input_node)) {
        // Do not guard with switch for monad inputs.
        continue;
      }
      // for some ops input should not guard it with switch
      if (InConvertWhiteList(node, index)) {
        continue;
      }

      // If the input for node is not the graph belonged, or it is an ValueNode.
      // Bypass the Primitive node which is inputs[0].
      if ((index >= 1 && input_node->func_graph() != nullptr && input_node->func_graph() != graph) ||
          ((index >= 1 && input_node->isa<ValueNode>()))) {
        input_node = generate_func(graph, cond, input_node);
        repl_node_inputs[std::pair<AnfNodePtr, size_t>(node, index)] = input_node;
        should_replace = true;
      }
      // Insert GeSwitch after load
      if (IsPrimitiveCNode(input_node, prim::kPrimLoad)) {
        auto cnode_load = input_node->cast<CNodePtr>();
        MS_EXCEPTION_IF_NULL(cnode_load);
        auto load_real_input = cnode_load->inputs()[kLoadRealInput];
        MS_EXCEPTION_IF_NULL(load_real_input);
        auto load_state_input = cnode_load->inputs()[kLoadStateInput];
        MS_EXCEPTION_IF_NULL(load_state_input);
        if ((load_real_input->func_graph() != nullptr && load_real_input->func_graph() != graph) &&
            !HasDependencyOnSubGraph(graph, load_state_input)) {
          input_node = generate_func(graph, cond, input_node);
          repl_node_inputs[std::pair<AnfNodePtr, size_t>(node, index)] = input_node;
          should_replace = true;
        }
      }

      if (input_node == nullptr) {
        MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, cond) << "generate switch node failed";
      }
    }
    if (should_replace) {
      auto new_node = graph->NewCNode({});
      repl_node[node] = new_node;
      (void)nodes_changed.emplace_back(node->cast<CNodePtr>(), new_node);
    }
  }
  RunSwitchNodeReplace(manager, nodes_changed, repl_node, repl_node_inputs, graph);
  return graph;
}

struct SharedOp {
  tensor::TensorPtr const_data;
  CNodePtr square_ops[2];
  CNodePtr merge_ops[2];
} MergeNetOutput;

inline tensor::TensorPtr GetConstData() { return MergeNetOutput.const_data; }
inline void SetConstData(const tensor::TensorPtr &const_value) { MergeNetOutput.const_data = const_value; }

inline CNodePtr GetSquareOp(int64_t switch_idx) { return MergeNetOutput.square_ops[switch_idx]; }
inline void SetSquareOp(int64_t switch_idx, const CNodePtr &op) { MergeNetOutput.square_ops[switch_idx] = op; }

inline CNodePtr GetMergeOp(int64_t switch_idx) { return MergeNetOutput.merge_ops[switch_idx]; }
inline void SetMergeOp(int64_t switch_idx, const CNodePtr &op) { MergeNetOutput.merge_ops[switch_idx] = op; }

inline void ResetSharedOp() {
  SetConstData(nullptr);
  SetSquareOp(0, nullptr);
  SetSquareOp(1, nullptr);
  SetMergeOp(0, nullptr);
  SetMergeOp(1, nullptr);
}

tensor::TensorPtr ConstData() {
  std::vector<int64_t> shp = {1};
  tensor::TensorPtr const_data = std::make_shared<tensor::Tensor>(kInt64->type_id(), shp);
  auto *val = static_cast<int64_t *>(const_data->data_c());
  *val = 0;
  return const_data;
}

CNodePtr SquareOp(const FuncGraphPtr &graph, const AnfNodePtr &cond, int64_t switch_idx,
                  const tensor::TensorPtr &const_data) {
  auto prim_square = prim::kPrimSquare;
  // for the depended node , add two const data to merge the flow ,one for depended node with same switch,
  // the other use the opposite
  auto ctrl_data = NewValueNode(const_data);
  auto ctrl_node = GenerateSwitchNode(graph, cond, ctrl_data, switch_idx);

  std::vector<AnfNodePtr> square_nodes{NewValueNode(prim_square), ctrl_node};
  auto square_op = graph->NewCNode(square_nodes);

  return square_op;
}

CNodePtr MergeNode(const FuncGraphPtr &graph, const AnfNodePtr &cond, int64_t switch_idx,
                   const tensor::TensorPtr &const_data, const CNodePtr &square_op) {
  // for the depended node , add two const data to merge the flow ,one for depended node with same switch,
  // the other use the opposite
  auto oppsite_ctrl_data = NewValueNode(const_data);
  auto opposite_ctrl_node = GenerateSwitchNode(graph, cond, oppsite_ctrl_data, 1 - switch_idx);

  std::vector<AnfNodePtr> merge_nodes;
  auto PrimMerge = prim::GetPythonOps("merge", "mindspore.ops.functional")->cast<PrimitivePtr>();
  merge_nodes.push_back(NewValueNode(PrimMerge));
  std::vector<AnfNodePtr> make_tuple_nodes{NewValueNode(prim::kPrimMakeTuple), square_op, opposite_ctrl_node};
  merge_nodes.push_back(graph->NewCNode(make_tuple_nodes));
  auto merge_op = graph->NewCNode(merge_nodes);

  return merge_op;
}

// merge(square_op(switch(ctrl_data)), switch(opposite_ctrl_data))
AnfNodePtr GenerateSwitchDependNode(const FuncGraphPtr &graph, const AnfNodePtr &cond, const AnfNodePtr &output_node,
                                    int64_t switch_idx) {
  tensor::TensorPtr const_data = GetConstData();
  if (const_data == nullptr) {
    const_data = ConstData();
    SetConstData(const_data);
  }

  CNodePtr square_op = GetSquareOp(switch_idx);
  if (square_op == nullptr) {
    square_op = SquareOp(graph, cond, switch_idx, const_data);
    SetSquareOp(switch_idx, square_op);
  }

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  AnfNodePtrList inputs = {NewValueNode(prim::kPrimDepend), square_op, output_node};
  auto depend_cnode = graph->NewCNode(inputs);
  if (!manager->Replace(square_op, depend_cnode)) {
    MS_LOG_WITH_NODE(EXCEPTION, square_op) << square_op->DebugString() << ", replace node failed.";
  }

  CNodePtr merge_op = GetMergeOp(switch_idx);
  if (merge_op == nullptr) {
    merge_op = MergeNode(graph, cond, switch_idx, const_data, square_op);
    SetMergeOp(switch_idx, merge_op);
  }

  return merge_op;
}

// generate switch nodes for true graph node inputs
AnfNodePtr GenerateSwitchDependTrueNode(const FuncGraphPtr &graph, const AnfNodePtr &cond, const AnfNodePtr &data) {
  // for switch op ,the output is a tuple ,0-th is false_branch, 1-th is true branch
  return GenerateSwitchDependNode(graph, cond, data, 1);
}

// generate switch nodes for false graph node inputs
AnfNodePtr GenerateSwitchDependFalseNode(const FuncGraphPtr &graph, const AnfNodePtr &cond, const AnfNodePtr &data) {
  // for switch op ,the output is a tuple ,0-th is false_branch, 1-th is true branch
  return GenerateSwitchDependNode(graph, cond, data, 0);
}

// to judge if the node used in Depend is a net output node
bool IsNetOutputNode(const FuncGraphManagerPtr &manager, const AnfNodePtr &node) {
  auto uses = manager->node_users()[node];
  bool is_output_node = true;
  for (auto &item : uses) {
    if (IsPrimitiveCNode(item.first, prim::kPrimDepend)) {
      continue;
    }
    is_output_node = false;
    break;
  }
  return is_output_node;
}

// generate node for Depended MakeTuple
void GenerateReplNodeForDependMakeTuple(
  const AnfNodePtr &depended_node, const FuncGraphPtr &graph, const AnfNodePtr &cond,
  const std::shared_ptr<mindspore::HashMap<AnfNodePtr, AnfNodePtr>> &repl_node,
  const std::function<AnfNodePtr(FuncGraphPtr graph, AnfNodePtr cond, AnfNodePtr data)> &generate_func) {
  MS_EXCEPTION_IF_NULL(graph->manager());

  auto depend_cnode = depended_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(depend_cnode);
  auto make_tuple_inputs = depend_cnode->inputs();
  const size_t make_tuple_begin_idx = 1;
  std::vector<AnfNodePtr> new_make_tuple_nodes;
  bool replace_make_tuple = false;
  new_make_tuple_nodes.push_back(NewValueNode(prim::kPrimMakeTuple));
  for (size_t idx = make_tuple_begin_idx; idx < make_tuple_inputs.size(); idx++) {
    auto depended_tuple_input_node = make_tuple_inputs[idx];
    if (IsPrimitiveCNode(depended_tuple_input_node->cast<CNodePtr>(), prim::kPrimDepend)) {
      new_make_tuple_nodes.push_back(depended_tuple_input_node);
      continue;
    }

    if (graph->manager()->node_users()[depended_tuple_input_node].size() == 1) {
      auto gen_node = generate_func(graph, cond, depended_tuple_input_node);
      new_make_tuple_nodes.push_back(gen_node);
      replace_make_tuple = true;
      continue;
    }

    MS_LOG(WARNING) << "depended node being used by others, ";
  }
  if (replace_make_tuple) {
    auto make_tuple_op = graph->NewCNode(new_make_tuple_nodes);
    (*repl_node)[depended_node] = make_tuple_op;
  }
}

// generate a replace depend node for a single network output node
void GenerateRepDepend(
  const CNodePtr &node, const FuncGraphPtr &graph, const AnfNodePtr &cond,
  const std::shared_ptr<mindspore::HashMap<AnfNodePtr, AnfNodePtr>> &repl_node,
  const std::function<AnfNodePtr(FuncGraphPtr graph, AnfNodePtr cond, AnfNodePtr data)> &generate_func) {
  MS_EXCEPTION_IF_NULL(graph->manager());

  auto inputs = node->inputs();
  if (inputs.size() != kDependInputSize) {
    MS_LOG_WITH_NODE(EXCEPTION, node)
      << "For 'Depend', the inputs should be [depend_prim, actual_value, depended_node].";
  }

  std::vector<AnfNodePtr> new_depened_inputs;
  // Inputs should be [depend, actual_value, depended_node]
  auto depended_node = inputs[kDependAttachNodeIndex];
  new_depened_inputs.push_back(inputs[0]);
  new_depened_inputs.push_back(inputs[1]);
  // depended node should be make_tuple or a single depended node
  if (IsPrimitiveCNode(depended_node, prim::kPrimMakeTuple)) {
    GenerateReplNodeForDependMakeTuple(depended_node, graph, cond, repl_node, generate_func);
  } else {
    // Check if there is only single user for depend_node.
    if (graph->manager()->node_users()[depended_node].size() == 1) {
      auto gen_node = generate_func(graph, cond, depended_node);
      (*repl_node)[depended_node] = gen_node;
    } else {
      MS_LOG(WARNING) << "depended node being used by others";
    }
  }
}

// generate depend node for netoutput node, to resolve the stream synchronize problem of ge
// traverse all nodes of depend node, find the graph output node , generaete a merge node of (square, const)
FuncGraphPtr TransformGraphDependNode(
  const FuncGraphPtr &graph, const AnfNodePtr &cond,
  const std::function<AnfNodePtr(FuncGraphPtr graph, AnfNodePtr cond, AnfNodePtr data)> &gen_depend_func) {
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  ResetSharedOp();
  std::shared_ptr<mindspore::HashMap<AnfNodePtr, AnfNodePtr>> repl_node =
    std::make_shared<mindspore::HashMap<AnfNodePtr, AnfNodePtr>>();  // record the node to be replaced
  const AnfNodeSet &nodes = graph->nodes();
  for (auto &node : nodes) {
    MS_EXCEPTION_IF_NULL(node);
    if (!node->isa<CNode>()) {
      continue;
    }
    if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
      auto cnode = node->cast<CNodePtr>();
      if (cnode->size() != kDependInputSize) {
        MS_LOG_WITH_NODE(EXCEPTION, cnode)
          << "For primitive 'Depend', the input size must be " << kDependInputSize << ", but got " << cnode->size();
      }
      auto depended_node = cnode->input(kDependAttachNodeIndex);
      MS_EXCEPTION_IF_NULL(depended_node);
      if (!depended_node->isa<CNode>()) {
        continue;
      }
      if (IsPrimitiveCNode(depended_node, prim::kPrimDepend)) {
        continue;
      }
      GenerateRepDepend(cnode, graph, cond, repl_node, gen_depend_func);
    }
  }
  ResetSharedOp();

  for (auto &item : *repl_node) {
    if (!manager->Replace(item.first, item.second)) {
      MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, item.first) << "TransformGraphDependNode replace node failed";
    }
  }

  return graph;
}

FuncGraphPtr TransformGraphCondTrueBranchNodes(const FuncGraphPtr &graph, const AnfNodePtr &cond) {
  (void)TransformGraphCondBranchNodes(graph, cond, GenerateSwitchTrueNode);
  return TransformGraphDependNode(graph, cond, GenerateSwitchDependTrueNode);
}

FuncGraphPtr TransformGraphCondFalseBranchNodes(const FuncGraphPtr &graph, const AnfNodePtr &cond) {
  (void)TransformGraphCondBranchNodes(graph, cond, GenerateSwitchFalseNode);
  return TransformGraphDependNode(graph, cond, GenerateSwitchDependFalseNode);
}

// Judge if the true and false graph output is compatible(they shall have same tuple size)
bool GraphOutputCompatible(const AbstractBasePtr &true_branch_abs, const AbstractBasePtr &false_branch_abs) {
  MS_EXCEPTION_IF_NULL(true_branch_abs);
  MS_EXCEPTION_IF_NULL(false_branch_abs);

  if (true_branch_abs->isa<abstract::AbstractTuple>() && false_branch_abs->isa<abstract::AbstractTuple>()) {
    abstract::AbstractTuplePtr true_branch_tuple = true_branch_abs->cast<abstract::AbstractTuplePtr>();
    abstract::AbstractTuplePtr false_branch_tuple = false_branch_abs->cast<abstract::AbstractTuplePtr>();
    if (true_branch_tuple->elements().size() != false_branch_tuple->elements().size()) {
      MS_LOG(ERROR) << "true branch size:" << true_branch_tuple->elements().size()
                    << ", not equal to false branch size:" << false_branch_tuple->elements().size() << " ";
      return false;
    }
    bool all_compatible = true;
    for (size_t i = 0; i < true_branch_tuple->elements().size(); i++) {
      all_compatible =
        all_compatible && GraphOutputCompatible(true_branch_tuple->elements()[i], false_branch_tuple->elements()[i]);
    }
    return all_compatible;
  }
  TypePtr true_branch_type = true_branch_abs->BuildType();
  TypePtr false_branch_type = false_branch_abs->BuildType();
  MS_LOG(DEBUG) << "branch output Type equal?" << (*true_branch_type == *false_branch_type)
                << " true:" << true_branch_type->ToString() << " false:" << false_branch_type->ToString();
  return (*true_branch_type == *false_branch_type);
}

// block_nodes[0]: condition node
// block_nodes[1]: true branch node
// block_nodes[2]: false branch node
// branch_output_abs[0]: true branch abstract
// branch_output_abs[1]: false branch abstract
AnfNodePtr GenerateMergeNodes(const std::vector<AnfNodePtr> &block_nodes,
                              const std::vector<AbstractBasePtr> &branch_output_abs, const FuncGraphPtr &switch_graph) {
  MS_EXCEPTION_IF_NULL(branch_output_abs[0]);
  MS_EXCEPTION_IF_NULL(branch_output_abs[1]);
  MS_EXCEPTION_IF_NULL(block_nodes[0]);
  MS_EXCEPTION_IF_NULL(switch_graph);
  auto PrimMerge = prim::GetPythonOps("merge", "mindspore.ops.functional")->cast<PrimitivePtr>();
  MS_EXCEPTION_IF_NULL(PrimMerge);

  if (!branch_output_abs[0]->isa<abstract::AbstractTuple>()) {
    std::vector<AnfNodePtr> merge_nodes;
    merge_nodes.push_back(NewValueNode(PrimMerge));
    std::vector<AnfNodePtr> make_tuple_nodes{NewValueNode(prim::kPrimMakeTuple), block_nodes[1], block_nodes[2]};
    merge_nodes.push_back(switch_graph->NewCNode(make_tuple_nodes));
    std::vector<AnfNodePtr> tuple_getitem_nodes{NewValueNode(prim::kPrimTupleGetItem),
                                                switch_graph->NewCNode(merge_nodes),
                                                NewValueNode(MakeValue(static_cast<int64_t>(0)))};
    return switch_graph->NewCNode(tuple_getitem_nodes);
  } else {
    auto true_branch_tuple = branch_output_abs[0]->cast<abstract::AbstractTuplePtr>();
    auto false_branch_tuple = branch_output_abs[1]->cast<abstract::AbstractTuplePtr>();
    MS_EXCEPTION_IF_NULL(true_branch_tuple);
    MS_EXCEPTION_IF_NULL(false_branch_tuple);

    std::vector<AnfNodePtr> make_tuple_nodes;
    make_tuple_nodes.push_back(NewValueNode(prim::kPrimMakeTuple));
    for (size_t i = 0; i < true_branch_tuple->elements().size(); i++) {
      std::vector<AnfNodePtr> true_getitem_nodes{NewValueNode(prim::kPrimTupleGetItem), block_nodes[1],
                                                 NewValueNode(MakeValue(SizeToLong(i)))};
      auto true_node = switch_graph->NewCNode(true_getitem_nodes);
      std::vector<AnfNodePtr> false_getitem_nodes{NewValueNode(prim::kPrimTupleGetItem), block_nodes[2],
                                                  NewValueNode(MakeValue(SizeToLong(i)))};
      auto false_node = switch_graph->NewCNode(false_getitem_nodes);

      auto merge_node = GenerateMergeNodes(
        {
          block_nodes[0],
          true_node,
          false_node,
        },
        {true_branch_tuple->elements()[i], false_branch_tuple->elements()[i]}, switch_graph);
      make_tuple_nodes.push_back(merge_node);
    }
    return switch_graph->NewCNode(make_tuple_nodes);
  }
}

AnfNodePtr TransformMergeBranches(const std::vector<AnfNodePtr> &block_nodes,
                                  const std::vector<AbstractBasePtr> &branch_output_abs,
                                  const FuncGraphPtr &func_graph) {
  if (!GraphOutputCompatible(branch_output_abs[0], branch_output_abs[1])) {
    MS_LOG_WITH_NODE(EXCEPTION, func_graph->return_node())
      << "Switch output branch not compatible, true:" << branch_output_abs[0]->ToString()
      << "， false:" << branch_output_abs[1]->ToString();
  }
  return GenerateMergeNodes(block_nodes, branch_output_abs, func_graph);
}
}  // namespace internal

bool ConvertSwitchReplacement::CheckSwitchBranch(const AnfNodePtr &node) const {
  if (!IsValueNode<FuncGraph>(node)) {
    return false;
  }
  // If graph contains FuncGraph, then ignore this node.
  auto graph = GetValueNode<FuncGraphPtr>(node);
  for (auto &item : graph->value_nodes()) {
    auto value_node = item.first;
    if (IsValueNode<FuncGraph>(value_node)) {
      return false;
    }
  }
  return true;
}

bool ConvertSwitchReplacement::CheckSwitchWrapNode(const AnfNodePtr &node) const {
  // {{prim::kPrimSwitch, X, G1, G2}, Xs}.
  if (node->isa<CNode>()) {
    auto inp0 = node->cast<CNodePtr>()->input(0);
    if (IsPrimitiveCNode(inp0, prim::kPrimSwitch)) {
      auto switch_node = inp0->cast<CNodePtr>();
      // for switch replace method, only graphs without graph inside can be replaced
      if (CheckSwitchBranch(switch_node->input(kTrueBranchIndex)) &&
          CheckSwitchBranch(switch_node->input(kFalseBranchIndex))) {
        return true;
      }
    }
  }
  return false;
}

void ConvertSwitchReplacement::TransformSwitchBranchReplace(const AnfNodePtr &node) const {
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  constexpr size_t input_index = 0;
  auto switch_cnode = cnode->input(input_index)->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(switch_cnode);
  auto cond = switch_cnode->input(kCondIndex);
  auto true_br = switch_cnode->input(kTrueBranchIndex);
  auto false_br = switch_cnode->input(kFalseBranchIndex);

  auto g1 = GetValueNode<FuncGraphPtr>(true_br);
  auto g2 = GetValueNode<FuncGraphPtr>(false_br);
  MS_EXCEPTION_IF_NULL(g1);
  MS_EXCEPTION_IF_NULL(g2);
  auto g1_output = g1->output();
  auto g2_output = g2->output();
  MS_EXCEPTION_IF_NULL(g1_output);
  MS_EXCEPTION_IF_NULL(g2_output);
  auto true_output = g1_output->abstract();
  auto false_output = g2_output->abstract();
  auto trans_g1 = internal::TransformGraphCondTrueBranchNodes(g1, cond);
  auto trans_g2 = internal::TransformGraphCondFalseBranchNodes(g2, cond);

  std::vector<AnfNodePtr> params;
  if (cnode->size() > 1) {
    // There are arguments for the call of switch result,
    // usually these are monad states added by auto-monad.
    for (size_t i = 1; i < cnode->size(); ++i) {
      params.push_back(cnode->inputs().at(i));
    }
  }
  auto fg = node->func_graph();
  auto cloned_g1 = InlineClone(trans_g1, fg, params);
  auto cloned_g2 = InlineClone(trans_g2, fg, params);
  auto new_node = internal::TransformMergeBranches({cond, cloned_g1, cloned_g2}, {true_output, false_output}, fg);
  (void)fg->manager()->Replace(node, new_node);
}

AnfNodePtr SwitchSimplify::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> cond;
  PatternNode<AnfNodePtr> true_br;
  PatternNode<AnfNodePtr> false_br;
  auto SwitchSimplLambda = [&node, &cond, &true_br, &false_br]() -> AnfNodePtr {
    auto value_ptr = GetValueNode(cond.GetNode(node));
    bool cond_value = false;
    if (value_ptr->isa<BoolImm>()) {
      cond_value = GetValue<bool>(value_ptr);
    } else {
      MS_LOG_WITH_NODE(EXCEPTION, node) << "The condition of branch must be a bool tensor value or a bool scalar value,"
                                        << " not support this condition value: " << value_ptr->ToString();
    }

    MS_LOG(DEBUG) << "condition value: " << value_ptr->ToString() << ", cond: " << cond_value
                  << ", node: " << node->DebugString();
    AnfNodePtr branch_node;
    if (cond_value) {
      branch_node = true_br.GetNode(node);
    } else {
      branch_node = false_br.GetNode(node);
    }
    auto fg = GetValuePtr<FuncGraph>(branch_node);
    if (fg != nullptr) {
      MS_LOG(DEBUG) << "No recursive, " << fg->ToString();
      fg->set_flag(FUNC_GRAPH_FLAG_NO_RECURSIVE, true);
    }
    return branch_node;
  };

  auto IsDeterminateCondition = [](const AnfNodePtr &node) -> bool { return IsValueNode<BoolImm>(node); };
  MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimSwitch, cond, true_br, false_br), SwitchSimplLambda,
                          cond.CheckFunc(IsDeterminateCondition, node));

  return nullptr;
}

AnfNodePtr CompareSwitchSimplify::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> cond;
  PatternNode<AnfNodePtr> true_br;
  PatternNode<AnfNodePtr> false_br;
  auto CompareSwitchSimplifyLambda = [&node, &cond, &true_br, &false_br]() -> AnfNodePtr {
    auto cnode = node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto compare_cnode = cnode->input(kIndex1)->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(compare_cnode);
    auto cond_tensor1 = GetValue<tensor::TensorPtr>(GetValueNode(compare_cnode->input(kIndex1)));
    auto cond_tensor2 = GetValue<tensor::TensorPtr>(GetValueNode(compare_cnode->input(kIndex2)));
    auto cond_value1 = reinterpret_cast<float *>(cond_tensor1->data_c());
    auto cond_value2 = reinterpret_cast<float *>(cond_tensor2->data_c());
    bool flag = false;
    if (IsPrimitiveCNode(compare_cnode, prim::kPrimLess) && (*cond_value1 < *cond_value2)) {
      flag = true;
    } else if (IsPrimitiveCNode(compare_cnode, prim::kPrimGreater) && (*cond_value1 > *cond_value2)) {
      flag = true;
    }
    if (flag) {
      return true_br.GetNode(node);
    }
    return false_br.GetNode(node);
  };

  auto ConstantCompareLambda = [](const AnfNodePtr &node) -> bool {
    if (!node->isa<CNode>()) {
      return false;
    }
    auto cnode = node->cast<CNodePtr>();
    if (!IsPrimitiveCNode(cnode, prim::kPrimLess) && !IsPrimitiveCNode(cnode, prim::kPrimGreater)) {
      return false;
    }
    bool has_no_value =
      std::any_of(cnode->inputs().begin() + kIndex1, cnode->inputs().end(), [](const AnfNodePtr &node) {
        if (!IsValueNode<tensor::Tensor>(node)) {
          return true;
        }
        auto value = GetValue<tensor::TensorPtr>(GetValueNode(node));
        if (value->device_address() != nullptr) {
          return true;
        }
        if (value->DataSize() > 1) {
          return true;
        }
        auto type_id = value->Dtype()->type_id();
        if (type_id != TypeId::kNumberTypeFloat32 && type_id != TypeId::kNumberTypeFloat) {
          return true;
        }
        return false;
      });
    return !has_no_value;
  };

  MATCH_REPLACE_LAMBDA_IF(node, PPrimitive(prim::kPrimSwitch, cond, true_br, false_br), CompareSwitchSimplifyLambda,
                          cond.CheckFunc(ConstantCompareLambda, node));

  return nullptr;
}

AnfNodePtr FloatTupleGetItemSwitch::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> cond;
  PatternNode<AnfNodePtr> true_br;
  PatternNode<AnfNodePtr> false_br;
  PatternNode<AnfNodePtr> x;
  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimTupleGetItem, PPrimitive(prim::kPrimSwitch, cond, true_br, false_br), x),
                   PPrimitive(prim::kPrimSwitch, cond, PPrimitive(prim::kPrimTupleGetItem, true_br, x),
                              PPrimitive(prim::kPrimTupleGetItem, false_br, x)),
                   x.CheckFunc(IsVNode, node));
  return nullptr;
}

AnfNodePtr FloatEnvironGetSwitch::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  PatternNode<AnfNodePtr> cond;
  PatternNode<AnfNodePtr> true_br;
  PatternNode<AnfNodePtr> false_br;
  PatternNode<AnfNodePtr> x;
  PatternNode<AnfNodePtr> x2;
  MATCH_REPLACE(node, PPrimitive(prim::kPrimEnvironGet, PPrimitive(prim::kPrimSwitch, cond, true_br, false_br), x, x2),
                PPrimitive(prim::kPrimSwitch, cond, PPrimitive(prim::kPrimEnvironGet, true_br, x, x2),
                           PPrimitive(prim::kPrimEnvironGet, false_br, x, x2)));

  return nullptr;
}

bool ConvertSwitchReplacement::operator()(const FuncGraphPtr &root, const OptimizerPtr &) const {
  auto manager = root->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto all_nodes = manager->all_nodes();

  bool change = false;
  for (auto &node : all_nodes) {
    if (CheckSwitchWrapNode(node)) {
      TransformSwitchBranchReplace(node);
      change = true;
    }
  }
  return change;
}

AnfNodePtr ExchangeSwitchDependValue::operator()(const OptimizerPtr &, const AnfNodePtr &node) {
  if (!node->isa<CNode>() || node->func_graph() == nullptr) {
    return nullptr;
  }
  ScopePtr scope = node->cast<CNodePtr>()->scope();
  ScopeGuard scope_guard(scope);

  PatternNode<AnfNodePtr> cond;
  PatternNode<AnfNodePtr> true_br;
  PatternNode<AnfNodePtr> false_br;
  PatternNode<AnfNodePtr> v;
  PatternNode<AnfNodePtr> x;
  MATCH_REPLACE_IF(node, PPrimitive(prim::kPrimSwitch, PPrimitive(prim::kPrimDepend, v, x), true_br, false_br),
                   PPrimitive(prim::kPrimDepend, PPrimitive(prim::kPrimSwitch, v, true_br, false_br), x),
                   IsVNode(v.GetNode(node)));
  return nullptr;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
