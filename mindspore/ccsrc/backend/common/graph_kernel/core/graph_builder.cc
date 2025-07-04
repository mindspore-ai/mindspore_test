/**
 * Copyright 2021-2024 Huawei Technologies Co., Ltd
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
#include "backend/common/graph_kernel/core/graph_builder.h"

#include <algorithm>
#include <memory>
#include <tuple>
#include <set>
#include <utility>
#include <vector>

#include "mindspore/ops/op_def/sequence_ops.h"
#include "ir/func_graph.h"
#include "include/common/utils/utils.h"
#include "utils/anf_utils.h"
#include "utils/ordered_set.h"
#include "backend/common/graph_kernel/core/graph_kernel_callback.h"
#include "backend/common/graph_kernel/core/graph_kernel_utils.h"
#include "backend/common/graph_kernel/graph_kernel_flags.h"
#include "ir/func_graph_cloner.h"
#include "backend/common/graph_kernel/core/value_depend_op_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "common/common_utils.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#ifdef ENABLE_AKG
#include "kernel/graph_kernel/graph_kernel_builder.h"
#endif

namespace mindspore::graphkernel {
// find outputs of nodes
AnfNodePtrList FindOutputs(const AnfNodePtrList &nodes, const AnfNodePtrToAnfNodePtrMap &eqv) {
  AnfNodePtrList output;
  auto mng = nodes[0]->func_graph()->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto &users = mng->node_users();
  for (auto &node : nodes) {
    // only CNode can be an output.
    if (!node->isa<CNode>()) {
      continue;
    }
    auto iter = users.find(node);
    if (iter == users.end()) {
      continue;
    }
    auto &node_users = iter->second;
    // if any user of the `node` is not in the nodes list, the `node` is an output.
    if (std::any_of(std::begin(node_users), std::end(node_users),
                    [&eqv](const std::pair<AnfNodePtr, int> &u) { return eqv.find(u.first) == eqv.end(); })) {
      (void)output.emplace_back(node);
    }
  }
  return output;
}

AnfNodePtr RefSubGraphNode(const FuncGraphPtr &fg, const AnfNodePtr &node, AnfNodePtrList *inputs_ptr,
                           AnfNodePtrToAnfNodePtrMap *eqv_ptr) {
  auto &eqv = *eqv_ptr;
  if (node->isa<ValueNode>() && !IsValueNode<FuncGraph>(node)) {
    eqv[node] = node;
  } else if (eqv.find(node) == eqv.end()) {
    inputs_ptr->push_back(node);
    eqv[node] = fg->add_parameter();
    eqv[node]->set_abstract(node->abstract());
    eqv[node]->set_kernel_info(node->kernel_info_ptr());
  }
  return eqv[node];
}

bool InlineInnerFuncGraph(const FuncGraphPtr &fg) {
  auto mng = fg->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  auto cnodes = fg->GetOrderedCnodes();
  for (const auto &n : cnodes) {
    auto graph_kernel_g = GetCNodeFuncGraph(n);
    if (graph_kernel_g == nullptr) {
      continue;
    }
    AnfNodePtrList inp(n->inputs().begin() + 1, n->inputs().end());
    auto out = InlineClone(graph_kernel_g, fg, inp, n);
    (void)mng->Replace(n, out);
    changed = true;
  }
  return changed;
}

void EliminateTupleOfTuple(const FuncGraphPtr &fg) {
  if (!IsPrimitiveCNode(fg->output(), prim::kPrimMakeTuple)) {
    return;
  }
  auto out_cnode = fg->output()->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(out_cnode);
  AnfNodePtrList new_args = GkUtils::SpreadTuples(out_cnode->inputs());
  if (new_args.size() != out_cnode->size()) {
    auto new_out = fg->NewCNode(new_args);
    auto mng = fg->manager();
    MS_EXCEPTION_IF_NULL(mng);
    (void)mng->Replace(out_cnode, new_out);
  }
  AbstractBasePtrList abs_list;
  (void)std::transform(new_args.begin() + 1, new_args.end(), std::back_inserter(abs_list),
                       [](const AnfNodePtr &node) { return node->abstract(); });
  fg->output()->set_abstract(std::make_shared<abstract::AbstractTuple>(abs_list));
}

template <typename T>
bool IsFinite(T value) {
  return !(std::isinf(value) || std::isnan(value));
}

bool IsFiniteScalar(void *data, TypeId type_id) {
  MS_EXCEPTION_IF_NULL(data);
  // check if float value is inf or nan
  if (type_id == kNumberTypeFloat64) {
    auto value = static_cast<double *>(data)[0];
    return IsFinite(value);
  } else if (type_id == kNumberTypeFloat32) {
    auto value = static_cast<float *>(data)[0];
    return IsFinite(value);
  } else if (type_id == kNumberTypeFloat16) {
    float16 *val = static_cast<float16 *>(data);
    auto value = static_cast<float>(val[0]);
    return IsFinite(value);
  }
  return true;
}

void UpdateBuildInfoOutputKernelObjectType(const AnfNodePtr &node) {
  if (node->kernel_info() == nullptr) {
    return;
  }
  auto build_info = AnfAlgo::GetSelectKernelBuildInfo(node);
  if (build_info != nullptr && build_info->GetAllOutputKernelObjectTypes().empty()) {
    auto abs_type = AnfAlgo::GetAbstractObjectType(node->abstract());
    auto object_type = kernel::TypeIdToKernelObjectType(abs_type);
    build_info->SetOutputsKernelObjectType(std::vector<kernel::KernelObjectType>{object_type});
  }
}

bool ConvertTensorToParameter(const FuncGraphPtr &fg, AnfNodePtrList *inputs_ptr) {
  auto cnodes = fg->GetOrderedCnodes();
  mindspore::OrderedSet<AnfNodePtr> value_nodes;
  for (const auto &cnode : cnodes) {
    auto &inputs = cnode->inputs();
    for (size_t i = 1; i < inputs.size(); ++i) {
      const auto &tnode = inputs[i];
      auto tensor = GetValueNode<tensor::TensorPtr>(tnode);
      if (tensor == nullptr) {
        continue;
      }
      auto primitive = GetCNodePrimitive(cnode);
      // For some primitives, the value in valuenode is required for further optimization.
      if (ValueDependOpUtils::KeepValueNode(primitive->name(), i - 1)) {
        continue;
      }
      auto type_id = tensor->data_type();
      // data is nullptr means uninitialized.
      if (tensor->data().const_data() == nullptr || tensor->DataSize() > 1 ||
          !IsFiniteScalar(tensor->data_c(), type_id) ||
          (type_id == kNumberTypeBool && GraphKernelFlags::GetInstance().kernel_generator == "DVM")) {
        (void)value_nodes.insert(tnode);
      }
    }
  }
  if (value_nodes.empty()) {
    return false;
  }
  auto mng = fg->manager();
  if (mng == nullptr) {
    mng = Manage(fg, false);
    fg->set_manager(mng);
  }
  for (const auto &vnode : value_nodes) {
    auto parameter = fg->add_parameter();
    parameter->set_abstract(vnode->abstract());
    parameter->set_kernel_info(vnode->kernel_info_ptr());
    UpdateBuildInfoOutputKernelObjectType(parameter);
    (void)mng->Replace(vnode, parameter);
    inputs_ptr->push_back(vnode);
  }
  return true;
}

bool SortParameters(const FuncGraphPtr &fg, AnfNodePtrList *inputs_ptr) {
  auto params = fg->parameters();
  if (params.size() != inputs_ptr->size()) {
    MS_LOG(EXCEPTION) << "parameters and inputs should have same size, but got " << params.size() << " and "
                      << inputs_ptr->size();
  }
  size_t n = inputs_ptr->size();
  using PairType = std::pair<AnfNodePtr, AnfNodePtr>;
  std::vector<PairType> normal_pairs;
  std::vector<PairType> monad_pairs;
  for (size_t i = 0; i < n; ++i) {
    if (HasAbstractMonad((*inputs_ptr)[i])) {
      (void)monad_pairs.emplace_back(params[i], (*inputs_ptr)[i]);
    } else {
      (void)normal_pairs.emplace_back(params[i], (*inputs_ptr)[i]);
    }
  }
  if (normal_pairs.empty() || monad_pairs.empty()) {
    return false;
  }
  auto normal_pairs_size = normal_pairs.size();
  for (size_t i = 0; i < normal_pairs_size; ++i) {
    params[i] = normal_pairs[i].first;
    (*inputs_ptr)[i] = normal_pairs[i].second;
  }
  for (size_t i = 0; i < monad_pairs.size(); ++i) {
    params[normal_pairs_size + i] = monad_pairs[i].first;
    (*inputs_ptr)[normal_pairs_size + i] = monad_pairs[i].second;
  }
  fg->set_parameters(std::move(params));
  return true;
}

bool IsTupleOutput(const AnfNodePtr &out, AnfNodePtrList *real_outs) {
  if (IsPrimitiveCNode(out, prim::kPrimMakeTuple)) {
    auto &inputs = out->cast<CNodePtr>()->inputs();
    real_outs->assign(inputs.begin() + 1, inputs.end());
    return true;
  }
  if (auto fg = GetCNodeFuncGraph(out); fg != nullptr) {
    return IsTupleOutput(fg->output(), real_outs);
  }
  return false;
}

void ReplaceNewFuseCNode(const FuncGraphPtr &func_graph, const AnfNodePtr &new_fuse_cnode,
                         const AnfNodePtrList &outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  // single out
  if (outputs.size() == 1) {
    (void)mng->Replace(outputs[0], new_fuse_cnode);
    return;
  }

  size_t offset = 0;
  for (size_t out_idx = 0; out_idx < outputs.size(); out_idx++) {
    AnfNodePtrList real_outs;
    // the output is a single tensor
    if (!IsTupleOutput(outputs[out_idx], &real_outs)) {
      auto gt_idx = MakeValue(SizeToLong(out_idx + offset));
      AnfNodePtrList gt_inputs{NewValueNode(prim::kPrimTupleGetItem), new_fuse_cnode, NewValueNode(gt_idx)};
      gt_inputs.back()->set_abstract(gt_idx->ToAbstract());
      auto new_out = func_graph->NewCNode(gt_inputs);
      new_out->set_abstract(outputs[out_idx]->abstract());
      (void)mng->Replace(outputs[out_idx], new_out);
      continue;
    }

    // the out is make tuple , modify the get_item node's value
    auto users = mng->node_users()[outputs[out_idx]];  // use a copy, the original user map is changed in for-loop.
    for (auto &user : users) {
      auto getitem_node = user.first;
      if (!getitem_node->isa<CNode>() || !IsPrimitiveCNode(getitem_node, prim::kPrimTupleGetItem)) {
        continue;
      }
      auto value_ptr = GetValueNode(getitem_node->cast<CNodePtr>()->input(kInputNodeOutputIndexInTupleGetItem));
      MS_EXCEPTION_IF_NULL(value_ptr);
      auto old_gt_idx = GetValue<int64_t>(value_ptr);
      auto gt_idx = MakeValue(SizeToLong(out_idx + offset) + old_gt_idx);
      AnfNodePtrList gt_inputs{NewValueNode(prim::kPrimTupleGetItem), new_fuse_cnode, NewValueNode(gt_idx)};
      gt_inputs.back()->set_abstract(gt_idx->ToAbstract());
      auto new_getitem_node = func_graph->NewCNode(gt_inputs);
      new_getitem_node->set_abstract(getitem_node->abstract());
      (void)mng->Replace(getitem_node, new_getitem_node);
    }

    offset += real_outs.size() - 1;
  }
}

// remove parameter which is not used
void EliminateRedundantParameters(const FuncGraphPtr &func_graph, AnfNodePtrList *inputs) {
  MS_EXCEPTION_IF_NULL(inputs);
  const auto &ori_parameter = func_graph->parameters();
  auto todos = TopoSort(func_graph->get_return());
  std::set<AnfNodePtr> used_param;
  for (auto node : todos) {
    if (node->isa<Parameter>()) {
      (void)used_param.insert(node);
    }
  }
  if (used_param.size() == ori_parameter.size()) {
    return;
  }
  AnfNodePtrList new_parameter;
  AnfNodePtrList new_inputs{(*inputs)[0]};
  for (size_t i = 0; i < ori_parameter.size(); ++i) {
    if (used_param.count(ori_parameter[i]) > 0) {
      new_parameter.push_back(ori_parameter[i]);
      new_inputs.push_back((*inputs)[i + 1]);
    }
  }
  func_graph->set_parameters(new_parameter);
  *inputs = std::move(new_inputs);
}

std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> BuildGraphFromNodes(const AnfNodePtrList &nodes) {
  return BuildGraphFromNodesInner(nodes, {});
}

#ifdef ENABLE_AKG
class RegGraphKernelBuilder {
 public:
  RegGraphKernelBuilder() { kernel::GraphKernelBuilder::build_func_ = BuildGraphFromNodes; }
} g_reg_graph_kernel_builder;
#endif

std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> BuildGraphFromNodesInner(const AnfNodePtrList &nodes,
                                                                                  const ClusterConfig &config) {
  FuncGraphPtr fg = nullptr;
  {
    // limit the lifetime of guard.
    TraceGuard guard(MakeTraceInfo<TraceSegmentTransform>(nodes[0]->cast<CNodePtr>()->func_graph()->debug_info()));
    fg = std::make_shared<FuncGraph>();
  }
  AnfNodePtrList input_list;
  AnfNodePtrToAnfNodePtrMap eqv;
  // Merge CNodes into a AnfGraph that represents a linear instruction segment
  for (auto &node : nodes) {
    if (IsPrimitiveCNode(node, prim::kPrimReturn)) {
      continue;
    }
    auto &node_inputs = node->cast<CNodePtr>()->inputs();
    std::vector<AnfNodePtr> new_args{node_inputs[0]};
    (void)std::transform(
      std::begin(node_inputs) + 1, std::end(node_inputs), std::back_inserter(new_args),
      [&fg, &input_list, &eqv](const AnfNodePtr &node) { return RefSubGraphNode(fg, node, &input_list, &eqv); });
    TraceGuard tg(MakeTraceInfo<TraceSegmentTransform>(node->debug_info()));
    eqv[node] = fg->NewCNode(new_args);
    eqv[node]->cast<CNodePtr>()->CloneCNodeInfo(node->cast<CNodePtr>());
    eqv[node]->cast<CNodePtr>()->set_fullname_with_scope(node->fullname_with_scope());
  }
  AnfNodePtrList outputs;
  if (config.only_output_basenode != nullptr) {
    // Make base node the only output of func_graph, to duplicate the overlapping parts
    if (eqv.find(config.only_output_basenode) == eqv.end()) {
      MS_LOG(EXCEPTION) << "Base node is not in the list of nodes: "
                        << config.only_output_basenode->fullname_with_scope();
    }
    outputs.push_back(config.only_output_basenode);
  } else {
    outputs = FindOutputs(nodes, eqv);
  }
  AnfNodePtr fg_output;
  if (outputs.size() > 1) {
    std::vector<AnfNodePtr> output_args;
    output_args.push_back(NewValueNode(prim::kPrimMakeTuple));
    (void)std::transform(std::begin(outputs), std::end(outputs), std::back_inserter(output_args),
                         [&eqv](const AnfNodePtr &o) -> AnfNodePtr { return eqv[o]; });
    AbstractBasePtrList out_abs_list;
    (void)std::transform(output_args.begin() + 1, output_args.end(), std::back_inserter(out_abs_list),
                         [](const AnfNodePtr &node) { return node->abstract(); });
    // Set output for AnfGraph
    fg_output = fg->NewCNode(output_args);
    fg_output->set_abstract(std::make_shared<abstract::AbstractTuple>(out_abs_list));
  } else {
    fg_output = eqv[outputs[0]];
  }
  fg->set_output(fg_output);
  return std::make_tuple(fg, input_list, outputs);
}

// Transform nodes(including basic and composite node) to a new graph, and collect their inputs and outputs.
std::tuple<FuncGraphPtr, AnfNodePtrList, AnfNodePtrList> BuildSingleGraphFromNodes(const AnfNodePtrList &nodes,
                                                                                   const ClusterConfig &config) {
  FuncGraphPtr fg;
  AnfNodePtrList inputs;
  AnfNodePtrList outputs;
  std::tie(fg, inputs, outputs) = BuildGraphFromNodesInner(nodes, config);

  FuncGraphManagerPtr mng = GkUtils::GetFuncGraphManager(fg);
  MS_EXCEPTION_IF_NULL(mng);

  if (config.inline_sub_func_graph) {
    (void)InlineInnerFuncGraph(fg);
  }
  // eliminate tuple of tuple, and set Abstract for output MakeTuple
  EliminateTupleOfTuple(fg);
  (void)EliminateMaketupleGetitem(fg);
  (void)ConvertTensorToParameter(fg, &inputs);
  if (config.sort_parameter) {
    SortParameters(fg, &inputs);
  }

  return std::make_tuple(fg, inputs, outputs);
}

CNodePtr CreateNewFuseCNode(const FuncGraphPtr &main_fg, const FuncGraphPtr &sub_fg, const AnfNodePtrList &inputs) {
  std::vector<AnfNodePtr> fn_inputs{NewValueNode(sub_fg)};
  (void)fn_inputs.insert(fn_inputs.end(), inputs.cbegin(), inputs.cend());
  EliminateRedundantParameters(sub_fg, &fn_inputs);
  auto fuse_cnode = main_fg->NewCNode(fn_inputs);
  fuse_cnode->set_abstract(sub_fg->output()->abstract());
  Callback::Instance()->SetGraphKernelNodeKernelInfo(fuse_cnode);
  return fuse_cnode;
}

CNodePtr ReplaceNodesWithGraphKernelNode(const AnfNodePtrList &nodes, const FuncGraphPtr &main_graph,
                                         const std::string &postfix, const ClusterConfig &config) {
  auto mng = main_graph->manager();
  if (mng == nullptr) {
    mng = Manage(main_graph, true);
    main_graph->set_manager(mng);
  }
  FuncGraphPtr fg;
  AnfNodePtrList inputs;
  AnfNodePtrList outputs;
  std::tie(fg, inputs, outputs) = BuildSingleGraphFromNodes(nodes, config);
  auto fuse_new_node = CreateNewFuseCNode(main_graph, fg, inputs);
  ReplaceNewFuseCNode(main_graph, fuse_new_node, outputs);
  auto fuse_op_name = GkUtils::ExtractGraphKernelName(nodes, "", postfix);
  fg->set_attr(FUNC_GRAPH_ATTR_GRAPH_KERNEL, MakeValue(fuse_op_name));
  return fuse_new_node;
}

// Eliminate redundant MakeTuple-Getitem edges
bool EliminateMaketupleGetitem(const FuncGraphPtr &fg) {
  auto nodes = fg->GetOrderedCnodes();
  auto mng = GkUtils::GetFuncGraphManager(fg);
  MS_EXCEPTION_IF_NULL(mng);
  bool changed = false;
  for (const auto &node : nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto gt = node->cast<CNodePtr>();
    auto mt = gt->input(kRealInputNodeIndexInTupleGetItem)->cast<CNodePtr>();
    if (mt == nullptr || !IsPrimitiveCNode(mt, prim::kPrimMakeTuple)) {
      continue;
    }
    auto idx = AnfUtils::GetIntValue(gt->input(kInputNodeOutputIndexInTupleGetItem));
    (void)mng->Replace(node, mt->input(LongToSize(idx + 1)));
    changed = true;
  }
  return changed;
}
}  // namespace mindspore::graphkernel
