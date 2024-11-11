/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "frontend/parallel/pass/interleave_split_concat_branches.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <set>
#include <string>
#include <queue>
#include <stack>
#include <unordered_map>

#include "mindspore/ops/op_def/array_op_name.h"
#include "mindspore/ops/op_def/math_op_name.h"
#include "mindspore/ops/op_def/other_op_name.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive.h"

#include "frontend/optimizer/optimizer.h"
#include "frontend/parallel/graph_util/graph_utils.h"
#include "include/common/utils/utils.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace parallel {
namespace {
auto const kEnableInterleave = "enable_interleave";
auto const kSplitConcatDepend = "split_concat_depend";
auto const kInterleaveBranchId = "interleave_branch_id";
auto const kInterleaveScopeId = "interleave_scope_id";
auto const kInterleaveSharedBranchId = 0;
auto const kGradFlag = "Gradients";
auto const kDefaultCostThreshold = 15;
auto const kFilterCost = 8;

struct InterLeaveScope {
  CNodePtr split{nullptr};
  CNodePtr concat{nullptr};
  bool forward{false};
  size_t scope_id{0};
  FuncGraphPtr graph{nullptr};
};

using InterLeaveScopePtr = std::shared_ptr<InterLeaveScope>;

struct BranchInterleaveNode {
  size_t begin{0};
  size_t end{0};
  float cost{0.0f};
  bool is_comm{false};
  bool overlapped{false};
  size_t branch_id{0};
};

using BranchInterleaveNodePtr = std::shared_ptr<BranchInterleaveNode>;

enum class InterleaveNodeType { kCommunication = 0, kComputation = 1, kVirtual = 2 };

inline bool IsForwardNode(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  if (!AnfUtils::IsRealKernel(node)) {
    return false;
  }
  return node->fullname_with_scope().find(kGradFlag) != 0;
}

// Branch id propagation to mask split independent branches
void PropagateBranchId(const InterLeaveScopePtr &interleave_scope, const CNodePtr &seed_node, size_t branch_id) {
  MS_EXCEPTION_IF_NULL(interleave_scope);
  MS_EXCEPTION_IF_NULL(seed_node);
  auto scope_id_value = MakeValue<size_t>(interleave_scope->scope_id);
  auto branch_id_value = MakeValue<size_t>(branch_id);
  static auto kSharedBranchIdValue = MakeValue<size_t>(kInterleaveSharedBranchId);
  seed_node->AddAttr(kInterleaveScopeId, scope_id_value);
  seed_node->AddAttr(kInterleaveBranchId, branch_id_value);
  std::queue<CNodePtr> to_visit;
  to_visit.emplace(seed_node);
  bool is_backward_scope = !interleave_scope->forward;
  while (!to_visit.empty()) {
    auto node = to_visit.front();
    to_visit.pop();
    MS_EXCEPTION_IF_NULL(node);
    if (node == interleave_scope->split) {
      continue;
    }

    for (auto &input : node->inputs()) {
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>()) {
        continue;
      }

      auto input_cnode = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_cnode);
      if (is_backward_scope && IsForwardNode(input_cnode)) {
        continue;
      }

      if (!input_cnode->HasAttr(kInterleaveScopeId)) {
        input_cnode->AddAttr(kInterleaveScopeId, scope_id_value);
      } else if (GetValue<size_t>(input_cnode->GetAttr(kInterleaveScopeId)) != interleave_scope->scope_id) {
        continue;
      }

      if (!input_cnode->HasAttr(kInterleaveBranchId)) {
        input_cnode->AddAttr(kInterleaveBranchId, branch_id_value);
        to_visit.emplace(input_cnode);
        continue;
      }

      auto input_branch_id = GetValue<size_t>(input_cnode->GetAttr(kInterleaveBranchId));
      if (input_branch_id != branch_id) {
        input_cnode->AddAttr(kInterleaveBranchId, kSharedBranchIdValue);
      }
    }
  }
}

mindspore::HashMap<CNodePtr, size_t> GetBranchNodesRefCount(const InterLeaveScopePtr &interleave_scope,
                                                            const CNodePtr &seed_node, size_t branch_id) {
  MS_EXCEPTION_IF_NULL(interleave_scope);
  MS_EXCEPTION_IF_NULL(seed_node);
  auto seen = NewSeenGeneration();
  mindspore::HashMap<CNodePtr, size_t> ref_count;
  std::queue<CNodePtr> to_visit;
  to_visit.emplace(seed_node);
  while (!to_visit.empty()) {
    auto node = to_visit.front();
    to_visit.pop();
    MS_EXCEPTION_IF_NULL(node);
    if (node == interleave_scope->split) {
      continue;
    }

    for (auto &input : node->inputs()) {
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>()) {
        continue;
      }

      auto input_cnode = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_cnode);
      if (input->seen_ == seen) {
        ref_count[input_cnode] += 1;
        continue;
      }

      if (!input_cnode->HasAttr(kInterleaveScopeId) ||
          GetValue<size_t>(input_cnode->GetAttr(kInterleaveScopeId)) != interleave_scope->scope_id) {
        continue;
      }

      if (!input_cnode->HasAttr(kInterleaveBranchId)) {
        continue;
      }

      auto input_branch_id = GetValue<size_t>(input_cnode->GetAttr(kInterleaveBranchId));
      if (input_branch_id == branch_id) {
        input->seen_ = seen;
        ref_count[input_cnode] = 1;
        to_visit.emplace(input_cnode);
      }
    }
  }
  return ref_count;
}

// Get ordered branch nodes
std::vector<CNodePtr> GetBranchOrderedNodes(const InterLeaveScopePtr &interleave_scope, const CNodePtr &seed_node,
                                            size_t branch_id) {
  MS_EXCEPTION_IF_NULL(interleave_scope);
  MS_EXCEPTION_IF_NULL(seed_node);
  auto seen = NewSeenGeneration();
  mindspore::HashMap<CNodePtr, size_t> ref_count = GetBranchNodesRefCount(interleave_scope, seed_node, branch_id);
  std::vector<CNodePtr> ordered_nodes;
  std::queue<CNodePtr> compute_queue;
  std::queue<CNodePtr> communication_queue;
  std::queue<CNodePtr> *current_queue = &compute_queue;
  compute_queue.emplace(seed_node);
  while (!compute_queue.empty() || !communication_queue.empty()) {
    if (current_queue->empty()) {
      current_queue = &compute_queue;
      if (compute_queue.empty()) {
        current_queue = &communication_queue;
      }
    }
    auto node = current_queue->front();
    current_queue->pop();
    MS_EXCEPTION_IF_NULL(node);
    if (AnfUtils::IsRealKernel(node)) {
      ordered_nodes.emplace_back(node);
    }

    for (auto &input : node->inputs()) {
      MS_EXCEPTION_IF_NULL(input);
      if (!input->isa<CNode>() || input->seen_ == seen) {
        continue;
      }

      auto input_cnode = input->cast<CNodePtr>();
      MS_EXCEPTION_IF_NULL(input_cnode);
      if (!input_cnode->HasAttr(kInterleaveScopeId) ||
          GetValue<size_t>(input_cnode->GetAttr(kInterleaveScopeId)) != interleave_scope->scope_id) {
        continue;
      }

      if (!input_cnode->HasAttr(kInterleaveBranchId)) {
        continue;
      }

      auto input_branch_id = GetValue<size_t>(input_cnode->GetAttr(kInterleaveBranchId));
      if (input_branch_id != branch_id) {
        continue;
      }

      auto iter_ref = ref_count.find(input_cnode);
      if (iter_ref == ref_count.end()) {
        continue;
      }

      if (iter_ref->second > 1) {
        --iter_ref->second;
        continue;
      }

      input->seen_ = seen;
      if (common::AnfAlgo::IsCommunicationOp(input_cnode)) {
        communication_queue.emplace(input_cnode);
      } else if (AnfUtils::IsRealKernel(input_cnode)) {
        compute_queue.emplace(input_cnode);
      } else {
        current_queue->emplace(input_cnode);
      }
    }
  }

  std::reverse(ordered_nodes.begin(), ordered_nodes.end());
  return ordered_nodes;
}

float GetNodeCost(const CNodePtr &node) {
  MS_EXCEPTION_IF_NULL(node);
  static const float kDefaultCost = 0.1f;
  static const std::unordered_map<std::string, float> kBaseCostMap = {
    {kAllReduceOpName, 8.0f},   {kReduceScatterOpName, 8.0f}, {kAllGatherOpName, 8.0f}, {kAllToAllOpName, 8.0f},
    {kAlltoAllOpName, 8.0f},    {kAllToAllvOpName, 8.0f},     {kAlltoAllVOpName, 8.0f}, {kReshapeOpName, 0.01f},
    {kBatchMatMulOpName, 8.0f}, {kMatMulOpName, 8.0f}};
  auto node_name = common::AnfAlgo::GetCNodeName(node);
  auto iter = kBaseCostMap.find(node_name);
  if (iter != kBaseCostMap.end()) {
    return iter->second;
  }

  return kDefaultCost;
}

// Interleave node is a range of the same compute type nodes
std::vector<BranchInterleaveNodePtr> GetInterleaveNodes(std::vector<CNodePtr> *node_vec_ptr, size_t branch_id) {
  MS_EXCEPTION_IF_NULL(node_vec_ptr);
  auto &node_vec = *node_vec_ptr;
  BranchInterleaveNodePtr current_node = nullptr;
  std::vector<BranchInterleaveNodePtr> result;
  auto current_node_type = InterleaveNodeType::kVirtual;
  for (size_t i = 0; i < node_vec.size(); ++i) {
    auto &node = node_vec[i];
    auto node_type = InterleaveNodeType::kVirtual;
    if (common::AnfAlgo::IsCommunicationOp(node)) {
      node_type = InterleaveNodeType::kCommunication;
    } else if (AnfUtils::IsRealKernel(node)) {
      node_type = InterleaveNodeType::kComputation;
    }

    if (node_type == InterleaveNodeType::kVirtual) {
      continue;
    }

    bool exceed_threshold = false;
    auto node_cost = GetNodeCost(node);
    if (current_node != nullptr && current_node->cost + node_cost > kDefaultCostThreshold) {
      exceed_threshold = true;
    }

    if (current_node != nullptr && (current_node_type != node_type || exceed_threshold)) {
      result.emplace_back(current_node);
      current_node = nullptr;
    }

    if (current_node == nullptr) {
      current_node_type = node_type;
      current_node = std::make_shared<BranchInterleaveNode>();
      current_node->branch_id = branch_id;
      current_node->begin = i;
      if (node_type == InterleaveNodeType::kCommunication) {
        current_node->is_comm = true;
      }
    }

    current_node->cost += node_cost;
    current_node->end = i;
  }

  if (current_node != nullptr) {
    result.emplace_back(current_node);
  }
  return result;
}

void AddDependNode(const FuncGraphPtr &graph, const CNodePtr &post_node, const AnfNodePtr &pre_node,
                   bool eager_depend = false) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(post_node);
  MS_EXCEPTION_IF_NULL(pre_node);
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto post_node_input = post_node->input(1);
  MS_EXCEPTION_IF_NULL(post_node_input);
  std::vector<AnfNodePtr> depend_inputs{NewValueNode(prim::kPrimDepend), post_node_input, pre_node};
  auto depend_node = graph->NewCNode(depend_inputs);
  depend_node->AddAttr(kSplitConcatDepend, MakeValue(1));
  if (eager_depend) {
    // eager visit depend in execute order builder
    depend_node->AddAttr(kAttrEagerDepend, MakeValue(1));
  }
  depend_node->set_abstract(post_node_input->abstract());
  manager->SetEdge(post_node, 1, depend_node);
}

void AddDependForOverlap(std::vector<std::vector<CNodePtr>> *branch_cnodes_ptr, const BranchInterleaveNodePtr &pre_node,
                         const BranchInterleaveNodePtr &cur_node, const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(branch_cnodes_ptr);
  MS_EXCEPTION_IF_NULL(pre_node);
  MS_EXCEPTION_IF_NULL(cur_node);
  MS_EXCEPTION_IF_NULL(graph);

  auto &branch_cnodes = *branch_cnodes_ptr;
  auto pre_node_branch_size = branch_cnodes[pre_node->branch_id].size();
  auto cur_node_branch_size = branch_cnodes[cur_node->branch_id].size();
  if (pre_node->begin >= pre_node_branch_size || pre_node->end >= pre_node_branch_size ||
      cur_node->begin >= cur_node_branch_size || cur_node->end >= cur_node_branch_size) {
    return;
  }

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto add_user_depend = [graph, manager](const CNodePtr &post_cnode, const CNodePtr &pre_cnode) {
    auto node_users = manager->node_users()[post_cnode];
    for (auto &node_user : node_users) {
      if (node_user.first == nullptr) {
        continue;
      }
      if (!node_user.first->isa<CNode>() || IsPrimitiveCNode(node_user.first, prim::kPrimPartial)) {
        continue;
      }
      AddDependNode(graph, node_user.first->cast<CNodePtr>(), pre_cnode);
    }
  };

  auto add_input_depend = [graph, manager](const CNodePtr &post_cnode, const CNodePtr &pre_cnode) {
    auto node_inputs = pre_cnode->inputs();
    for (auto &node_input : node_inputs) {
      if (node_input == nullptr) {
        continue;
      }
      if (!node_input->isa<CNode>()) {
        continue;
      }
      AddDependNode(graph, post_cnode, node_input);
    }
  };

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto lhs_node = pre_node;
  auto rhs_node = cur_node;
  if (!pre_node->is_comm) {
    lhs_node = cur_node;
    rhs_node = pre_node;
  }

  auto const &lhs_cnodes = branch_cnodes[lhs_node->branch_id];
  auto const &rhs_cnodes = branch_cnodes[rhs_node->branch_id];
  auto lhs_begin_cnode = lhs_cnodes[lhs_node->begin];
  auto lhs_end_cnode = lhs_cnodes[lhs_node->end];
  auto rhs_begin_cnode = rhs_cnodes[rhs_node->begin];
  auto rhs_end_cnode = rhs_cnodes[rhs_node->end];

  // 1.Avoid communication users node block comm/comp parallel
  // 2.Insert communication nodes before computation nodes for kbk
  // lhs-begin      <----    rhs-allnode-input
  // lhs-end-users  <----    rhs-end
  // lhs-end        ----->   rhs-begin       for kbk
  // lhs-end        ----->   rhs-begin       for ge

  if (ms_context->IsKByKExecutorMode()) {
    auto mm_begin = rhs_node->begin;
    while (mm_begin <= rhs_node->end && GetNodeCost(rhs_cnodes[mm_begin]) < kFilterCost) {
      ++mm_begin;
    }
    if (mm_begin > rhs_node->end) {
      mm_begin = rhs_node->begin;
    }
    rhs_begin_cnode = rhs_cnodes[mm_begin];
    add_input_depend(lhs_begin_cnode, rhs_begin_cnode);
    add_user_depend(lhs_end_cnode, rhs_end_cnode);
    AddDependNode(graph, rhs_begin_cnode, lhs_end_cnode, true);
  } else {
    add_input_depend(lhs_begin_cnode, rhs_begin_cnode);
    add_user_depend(lhs_end_cnode, rhs_end_cnode);
    AddDependNode(graph, rhs_begin_cnode, lhs_begin_cnode->input(1));
  }
}

// Interleave two branches nodes to overlap comm/comp
// Return unoverlapped nodes
std::vector<BranchInterleaveNodePtr> InterleaveTwoBranch(std::vector<std::vector<CNodePtr>> *branch_cnodes_ptr,
                                                         std::vector<BranchInterleaveNodePtr> *pre_nodes_ptr,
                                                         std::vector<BranchInterleaveNodePtr> *cur_nodes_ptr,
                                                         const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(pre_nodes_ptr);
  MS_EXCEPTION_IF_NULL(cur_nodes_ptr);
  auto &pre_nodes = *pre_nodes_ptr;
  auto &cur_nodes = *cur_nodes_ptr;
  if (pre_nodes.empty()) {
    return cur_nodes;
  }

  if (cur_nodes.empty()) {
    return pre_nodes;
  }

  std::vector<BranchInterleaveNodePtr> unoverlapped_nodes;
  size_t j = 0;
  size_t k = 0;
  for (size_t i = 0; i < cur_nodes.size(); ++i) {
    auto &cur_node = cur_nodes[i];
    MS_EXCEPTION_IF_NULL(cur_node);
    if (cur_node->cost < kFilterCost) {
      cur_node->overlapped = true;
      continue;
    }

    j = k;
    if (j == pre_nodes.size()) {
      if (!cur_node->overlapped) {
        (void)unoverlapped_nodes.emplace_back(cur_node);
      }
      continue;
    }

    bool get_next = false;
    while (j < pre_nodes.size()) {
      auto &pre_node = pre_nodes[j];
      MS_EXCEPTION_IF_NULL(pre_node);
      if (get_next) {
        if (pre_node->cost >= kFilterCost) {
          break;
        }
        pre_node->overlapped = true;
        ++j;
        k = j;
        continue;
      }

      if (pre_node->cost < kFilterCost) {
        pre_node->overlapped = true;
        ++j;
      } else if (cur_node->is_comm == pre_node->is_comm) {
        ++j;
      } else {
        AddDependForOverlap(branch_cnodes_ptr, pre_node, cur_node, graph);
        cur_node->overlapped = true;
        pre_node->overlapped = true;
        for (auto check_index = k; check_index < j; ++check_index) {
          auto &check_node = pre_nodes[check_index];
          if (!check_node->overlapped) {
            (void)unoverlapped_nodes.emplace_back(check_node);
          }
        }
        ++j;
        k = j;
        get_next = true;
      }
    }
    if (!cur_node->overlapped) {
      (void)unoverlapped_nodes.emplace_back(cur_node);
    }
  }

  return unoverlapped_nodes;
}

void PrintBranchNodes(const std::vector<CNodePtr> &branch_nodes) {
  MS_LOG(INFO) << "Start print split concat interleave branch node name:";
  for (auto &node : branch_nodes) {
    if (node == nullptr) {
      continue;
    }
    MS_LOG(INFO) << node->fullname_with_scope();
  }
  MS_LOG(INFO) << "End print split concat interleave branch node name.";
}

void InterleaveBranchesForCommunicationOverlap(const InterLeaveScopePtr &interleave_scope) {
  MS_EXCEPTION_IF_NULL(interleave_scope);
  MS_EXCEPTION_IF_NULL(interleave_scope->concat);
  MS_EXCEPTION_IF_NULL(interleave_scope->split);
  auto make_tuple = interleave_scope->concat->input(1);
  MS_EXCEPTION_IF_NULL(make_tuple);
  auto make_tuple_cnode = make_tuple->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(make_tuple_cnode);
  auto branch_outputs = make_tuple_cnode->inputs();

  // Find independent branch nodes
  std::vector<std::vector<CNodePtr>> total_branch_nodes;
  size_t interleave_branch_num = branch_outputs.size() - 1;
  if (interleave_branch_num <= 1) {
    return;
  }

  for (size_t i = 0; i < interleave_branch_num; ++i) {
    auto branch_id = i + 1;
    auto &branch_output = branch_outputs[branch_id];
    auto branch_output_cnode = branch_output->cast<CNodePtr>();
    if (branch_output_cnode == nullptr) {
      return;
    }
    PropagateBranchId(interleave_scope, branch_output_cnode, branch_id);
  }

  // Get ordered branch nodes
  for (size_t i = 0; i < interleave_branch_num; ++i) {
    auto branch_id = i + 1;
    auto &branch_output = branch_outputs[branch_id];
    auto branch_output_cnode = branch_output->cast<CNodePtr>();
    if (branch_output_cnode == nullptr) {
      return;
    }
    auto branch_nodes = GetBranchOrderedNodes(interleave_scope, branch_output_cnode, branch_id);
    total_branch_nodes.emplace_back(branch_nodes);
    PrintBranchNodes(branch_nodes);
  }

  size_t cur_index = 1;
  auto pre_nodes = GetInterleaveNodes(&total_branch_nodes[0], 0);
  while (cur_index < total_branch_nodes.size()) {
    auto cur_nodes = GetInterleaveNodes(&total_branch_nodes[cur_index], cur_index);
    // Interleave two branches nodes to overlap comm/comp
    pre_nodes = InterleaveTwoBranch(&total_branch_nodes, &pre_nodes, &cur_nodes, interleave_scope->graph);
    ++cur_index;
  }
}
}  // namespace

void InterleaveSplitConcatBranches(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_enable = ms_context->get_param<bool>(MS_CTX_ENABLE_INTERLEAVE_SPLIT_CONCAT_BRANCH);
  if (!is_enable) {
    return;
  }

  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<InterLeaveScopePtr> interleave_scopes;

  // Use unique id set to find correspond backward ops
  std::set<std::string> target_unique_id_set;
  size_t scope_id = 0;
  for (const auto &child_graph : manager->func_graphs()) {
    auto graph_orders = child_graph->GetOrderedCnodes();
    std::vector<CNodePtr> origin_nodes_topological(graph_orders.cbegin(), graph_orders.cend());
    InterLeaveScopePtr current_scope = nullptr;
    std::stack<InterLeaveScopePtr> scope_stack;
    for (const auto &node : origin_nodes_topological) {
      MS_EXCEPTION_IF_NULL(node);
      // Only split and concat node is the target node
      bool is_split = IsPrimitiveCNode(node, prim::kPrimSplit);
      bool is_concat = IsPrimitiveCNode(node, prim::kPrimConcat);
      if (!is_split && !is_concat) {
        continue;
      }

      bool forward = true;
      auto prim = common::AnfAlgo::GetCNodePrimitive(node);
      MS_EXCEPTION_IF_NULL(prim);
      // Node with enable interleave flag or its backward node
      if (!prim->HasAttr(kEnableInterleave)) {
        if (!node->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
          continue;
        }
        if (target_unique_id_set.find(GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrForwardUniqueId))) ==
            target_unique_id_set.end()) {
          continue;
        }
        forward = false;
      }

      if (forward && node->HasPrimalAttr(kPrimalAttrUniqueId)) {
        (void)target_unique_id_set.emplace(GetValue<std::string>(node->GetPrimalAttr(kPrimalAttrUniqueId)));
      }

      if (is_split) {
        current_scope = std::make_shared<InterLeaveScope>();
        current_scope->graph = child_graph;
        current_scope->split = node;
        current_scope->forward = forward;
        current_scope->scope_id = ++scope_id;
        scope_stack.push(current_scope);
      }

      if (is_concat) {
        if (scope_stack.empty()) {
          continue;
        }
        current_scope = scope_stack.top();
        current_scope->concat = node;
        interleave_scopes.emplace_back(current_scope);
        scope_stack.pop();
      }
    }
  }

  // Handle each split/concat scope
  for (auto &interleave_scope : interleave_scopes) {
    InterleaveBranchesForCommunicationOverlap(interleave_scope);
  }
}
}  // namespace parallel
}  // namespace mindspore
