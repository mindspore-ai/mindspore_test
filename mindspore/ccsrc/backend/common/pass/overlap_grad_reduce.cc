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
#include "backend/common/pass/overlap_grad_reduce.h"

#include <memory>
#include <queue>
#include <vector>
#include <utility>
#include <algorithm>

#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "ir/graph_utils.h"
#include "include/common/utils/utils.h"
#include "include/common/utils/parallel_context.h"
#include "mindspore/core/include/utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
constexpr char kAccuGradsPrefix[] = "accu_grads.";
typedef struct GradReduceUser {
 public:
  GradReduceUser() {
    assign_add_list.clear();
    grad_reduce_list.clear();
    latest_dw_execute_order = 0;
    latest_dw_compute_node = nullptr;
  }
  GradReduceUser(const CNodePtrList &assign_add_list_, const CNodePtrList &grad_reduce_list_,
                 const CNodePtr &latest_dw_compute_node, size_t latest_execute_order_)
      : assign_add_list(assign_add_list_),
        grad_reduce_list(grad_reduce_list_),
        latest_dw_compute_node(latest_dw_compute_node),
        latest_dw_execute_order(latest_execute_order_) {}

  CNodePtrList assign_add_list;
  CNodePtrList grad_reduce_list;
  AnfNodePtr latest_dw_compute_node;
  size_t latest_dw_execute_order;
} GradReduceUser;

bool IsFromParallelOptimizerRs(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimReduceScatter)) {
    return false;
  }
  auto prim = GetCNodePrimitive(node->cast<CNodePtr>());
  if (prim->instance_name().find("grad_parallel_optimizer") == std::string::npos) {
    return false;
  }
  return true;
}

bool IsFromGradMirrorAR(const AnfNodePtr &node) {
  if (!IsPrimitiveCNode(node, prim::kPrimAllReduce)) {
    return false;
  }
  auto prim = GetCNodePrimitive(node->cast<CNodePtr>());
  if (prim->instance_name().find("grad_mirror") == std::string::npos) {
    return false;
  }
  return true;
}

CNodePtr CreateMakeTupleNode(const FuncGraphPtr &func_graph, const AnfNodePtrList &tuple_inputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  AnfNodePtrList new_make_tuple_inputs = {NewValueNode(prim::kPrimMakeTuple)};
  (void)new_make_tuple_inputs.insert(new_make_tuple_inputs.cend(), tuple_inputs.cbegin(), tuple_inputs.cend());
  auto make_tuple_node = func_graph->NewCNode(new_make_tuple_inputs);
  MS_EXCEPTION_IF_NULL(make_tuple_node);

  // MakeTuple's abstract must consist of all inputs' abstract in case unexpected graph compiling error.
  AbstractBasePtrList abstract_list;
  (void)std::for_each(tuple_inputs.cbegin(), tuple_inputs.cend(),
                      [&](const auto &input) { (void)abstract_list.emplace_back(input->abstract()); });
  if (std::find_if(abstract_list.begin(), abstract_list.end(), [](auto abs) { return !abs; }) != abstract_list.end()) {
    return make_tuple_node;
  }
  make_tuple_node->set_abstract(std::make_shared<abstract::AbstractTuple>(abstract_list));
  return make_tuple_node;
}

void InsertDepend(const AnfNodePtr &prior_node, const AnfNodePtr &post_node, const FuncGraphManagerPtr &manager,
                  const FuncGraphPtr &root, const std::string &attr_tag = "", const size_t post_node_input_index = 1) {
  MS_EXCEPTION_IF_NULL(prior_node);
  MS_EXCEPTION_IF_NULL(post_node);
  auto post_cnode = post_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(post_cnode);
  std::vector<AnfNodePtr> depend_input = {NewValueNode(prim::kPrimDepend), post_cnode->input(post_node_input_index),
                                          prior_node};
  auto depend_node = root->NewCNode(depend_input);
  depend_node->set_abstract(post_cnode->input(post_node_input_index)->abstract());
  if (!attr_tag.empty()) {
    depend_node->AddAttr(attr_tag, MakeValue<bool>(true));
  }
  manager->SetEdge(post_node, post_node_input_index, depend_node);
}

std::optional<std::string> GetRefKeyFromNode(const AnfNodePtr &node) {
  auto abs = node->abstract();
  if (abs == nullptr) {
    // Abstract for some depend node are not proper set, we follow its input.
    if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
      return GetRefKeyFromNode(node->cast<CNodePtr>());
    }
    // Abstract should be set except UpdateState nodes.
    if (!IsPrimitiveCNode(node, prim::kPrimUpdateState)) {
      MS_LOG(WARNING) << "Abstract not set for " << node->DebugString();
    }
    return std::nullopt;
  }
  auto abs_ref = abs->cast<abstract::AbstractRefPtr>();
  if (abs_ref == nullptr) {
    return std::nullopt;
  }
  auto ref_key = abs_ref->ref_key_value()->cast<StringImmPtr>();
  if (ref_key == nullptr) {
    return std::nullopt;
  }
  return ref_key->value();
}

std::optional<std::string> GetMirrorUserIdFromCNode(const CNodePtr &cnode) {
  if (!cnode->HasPrimalAttr(kPrimalAttrMirrorUserId)) {
    return std::nullopt;
  }
  auto mirror_user_id = GetValue<std::string>(cnode->GetPrimalAttr(kPrimalAttrMirrorUserId));
  return mirror_user_id;
}

std::vector<std::pair<AnfNodePtr, int>> GetOutputNodesWithFilter(const AnfNodePtr &node,
                                                                 std::function<bool(const AnfNodePtr &)> filter) {
  auto func_graph = node->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  std::vector<std::pair<AnfNodePtr, int>> res;
  std::queue<AnfNodePtr> anf_queue;
  anf_queue.push(node);
  while (!anf_queue.empty()) {
    auto queue_end = anf_queue.front();
    anf_queue.pop();
    auto user_set = manager->node_users()[queue_end];
    for (auto &pair : user_set) {
      if (filter(pair.first)) {
        anf_queue.push(pair.first);
        continue;
      }
      res.push_back(pair);
    }
  }
  return res;
}

AnfNodePtr GetDwComputeNode(const CNodePtr &cnode, size_t input_index) {
  MS_EXCEPTION_IF_NULL(cnode);
  AnfNodePtr dw_compute_node = cnode->input(input_index);
  while (IsOneOfPrimitiveCNode(dw_compute_node, {prim::kPrimDepend, prim::kPrimCast, prim::kPrimTupleGetItem})) {
    auto dw_compute_cnode = dw_compute_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(dw_compute_cnode);
    dw_compute_node = dw_compute_cnode->input(kIndex1);
  }
  return dw_compute_node;
}

CNodePtr FindDxMatMulByDw(const AnfNodePtr &dw_node) {
  MS_EXCEPTION_IF_NULL(dw_node);
  if (!IsOneOfPrimitiveCNode(dw_node, {prim::kPrimMatMul, prim::kPrimMatMulExt}) ||
      !dw_node->cast<CNodePtr>()->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
    return nullptr;
  }
  auto dw_cnode = dw_node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(dw_cnode);
  auto dw_prim = GetCNodePrimitive(dw_cnode);
  auto dw_forward_unique_id = GetValue<std::string>(dw_cnode->GetPrimalAttr(kPrimalAttrForwardUniqueId));
  AnfNodePtr common_input = dw_cnode->input(kIndex1);
  while (IsPrimitiveCNode(common_input, prim::kPrimDepend)) {
    common_input = common_input->cast<CNodePtr>()->input(kIndex1);
  }
  auto common_input_user_list = GetOutputNodesWithFilter(
    common_input, [](const AnfNodePtr &node) { return IsOneOfPrimitiveCNode(node, {prim::kPrimDepend}); });
  CNodePtr dx_cnode = nullptr;
  for (const auto &user_pair : common_input_user_list) {
    auto user_node = user_pair.first;
    if (!IsPrimitiveCNode(user_node, dw_prim) || user_node == dw_cnode) {
      continue;
    }
    auto user_cnode = user_node->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user_cnode);
    if (!user_cnode->HasPrimalAttr(kPrimalAttrForwardUniqueId)) {
      continue;
    }
    if (GetValue<std::string>(user_cnode->GetPrimalAttr(kPrimalAttrForwardUniqueId)) == dw_forward_unique_id) {
      return user_cnode;
    }
  }
  return nullptr;
}

std::unordered_map<std::string, CNodePtrList> ExtractAssignAddByMirrorUser(
  const CNodePtrList &execute_order_cnode_list) {
  std::unordered_map<std::string, CNodePtrList> assign_add_map;
  for (const CNodePtr &cur_cnode : execute_order_cnode_list) {
    if (!IsPrimitiveCNode(cur_cnode, prim::kPrimAssignAdd)) {
      continue;
    }
    auto ref_key = GetRefKeyFromNode(cur_cnode->input(kIndex1));
    if (!ref_key.has_value() || ref_key.value().find(kAccuGradsPrefix) != kIndex0) {
      continue;
    }
    if (assign_add_map.find(ref_key.value()) == assign_add_map.end()) {
      assign_add_map[ref_key.value()] = CNodePtrList{cur_cnode};
    } else {
      assign_add_map[ref_key.value()].push_back(cur_cnode);
    }
  }
  return assign_add_map;
}

std::unordered_map<std::string, GradReduceUser> ExtractGradReduceByMirrorUser(
  const CNodePtrList &execute_order_cnode_list) {
  std::unordered_map<std::string, GradReduceUser> grad_reduce_map;
  for (const CNodePtr &cur_cnode : execute_order_cnode_list) {
    if (!IsFromParallelOptimizerRs(cur_cnode) && !IsFromGradMirrorAR(cur_cnode)) {
      continue;
    }
    auto mirror_user = GetMirrorUserIdFromCNode(cur_cnode);
    if (!mirror_user.has_value()) {
      continue;
    }
    if (grad_reduce_map.find(mirror_user.value()) == grad_reduce_map.end()) {
      auto grad_reduce_user = GradReduceUser();
      grad_reduce_user.grad_reduce_list.push_back(cur_cnode);
      grad_reduce_map[mirror_user.value()] = grad_reduce_user;
    } else {
      grad_reduce_map[mirror_user.value()].grad_reduce_list.push_back(cur_cnode);
    }
  }
  return grad_reduce_map;
}

std::vector<GradReduceUser> ExtractGradReduceUserList(const CNodePtrList &execute_order_cnode_list,
                                                      bool with_accumulation) {
  auto grad_reduce_map = ExtractGradReduceByMirrorUser(execute_order_cnode_list);

  if (with_accumulation) {
    auto assign_add_map = ExtractAssignAddByMirrorUser(execute_order_cnode_list);
    for (auto grad_reduce_user_pair : grad_reduce_map) {
      auto mirror_user_id = grad_reduce_user_pair.first;
      auto expect_accu_grad_ref_key = kAccuGradsPrefix + mirror_user_id;
      if (assign_add_map.find(expect_accu_grad_ref_key) == assign_add_map.end()) {
        MS_LOG(EXCEPTION) << "Cannot find accu_grad '" << expect_accu_grad_ref_key << "' in assign_add_map";
      }
      grad_reduce_map[mirror_user_id].assign_add_list = assign_add_map[expect_accu_grad_ref_key];
    }
  }
  std::vector<GradReduceUser> ret;
  std::transform(grad_reduce_map.begin(), grad_reduce_map.end(), std::back_inserter(ret),
                 [](const std::pair<std::string, GradReduceUser> &pair) { return pair.second; });
  for (auto &grad_reduce_user : ret) {
    AnfNodePtr latest_dw_compute_node;
    if (grad_reduce_user.assign_add_list.empty()) {
      latest_dw_compute_node = GetDwComputeNode(grad_reduce_user.grad_reduce_list.front(), kIndex1);
      MS_EXCEPTION_IF_NULL(latest_dw_compute_node);
    } else {
      latest_dw_compute_node = GetDwComputeNode(grad_reduce_user.assign_add_list.back(), kIndex2);
      MS_EXCEPTION_IF_NULL(latest_dw_compute_node);
    }
    auto latest_dw_execution_order =
      std::find(execute_order_cnode_list.begin(), execute_order_cnode_list.end(), latest_dw_compute_node) -
      execute_order_cnode_list.begin();
    grad_reduce_user.latest_dw_compute_node = latest_dw_compute_node;
    grad_reduce_user.latest_dw_execute_order = latest_dw_execution_order;
  }
  std::sort(ret.begin(), ret.end(), [](const GradReduceUser &a, const GradReduceUser &b) {
    return a.latest_dw_execute_order < b.latest_dw_execute_order;
  });
  return ret;
}
}  // namespace

bool OverlapGradReduce::DoOverlapGradReduce(const KernelGraphPtr &kernel_graph, bool with_accumulation) {
  const auto &execution_order = kernel_graph->execution_order();
  auto grad_reduce_user_list = ExtractGradReduceUserList(execution_order, with_accumulation);
  if (grad_reduce_user_list.empty()) {
    MS_LOG(DEBUG) << "grad reduce_user_list is empty, skip it.";
    return false;
  }
  auto manager = kernel_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);

  auto last_grad_reduce_node = grad_reduce_user_list.back().grad_reduce_list.back();
  for (const auto &grad_reduce_user : grad_reduce_user_list) {
    // Reconstruct depend for grad reduce
    auto next_op_users = manager->node_users()[grad_reduce_user.grad_reduce_list.back()];
    if (with_accumulation) {
      auto assign_add_list = grad_reduce_user.assign_add_list;
      auto grad_reduce_list = grad_reduce_user.grad_reduce_list;
      if (assign_add_list.size() > 1) {
        AnfNodePtrList inputs(assign_add_list.begin(), assign_add_list.end());
        auto make_tuple_cnode = CreateMakeTupleNode(kernel_graph, inputs);
        MS_EXCEPTION_IF_NULL(make_tuple_cnode);
        auto accu_input = grad_reduce_list.front()->input(kIndex1);
        while (IsPrimitiveCNode(accu_input, prim::kPrimDepend)) {
          accu_input = accu_input->cast<CNodePtr>()->input(kIndex1);
        }
        auto depend_cnode = kernel_graph->NewCNode({NewValueNode(prim::kPrimDepend), accu_input, make_tuple_cnode});
        depend_cnode->set_abstract(accu_input->abstract());
        depend_cnode->AddAttr("grad_comm_assign_add_depend", MakeValue<bool>(true));
        manager->SetEdge(grad_reduce_list.front(), kIndex1, depend_cnode);
      } else {
        InsertDepend(assign_add_list.front(), grad_reduce_list.front(), manager, kernel_graph,
                     "grad_comm_assign_add_depend");
      }
    }
    // Move all communication users to the back of the last gradient communication.
    for (const auto &next_op_user : next_op_users) {
      InsertDepend(last_grad_reduce_node, next_op_user.first, manager, kernel_graph, "last_grad_comm_compute_depend");
    }
  }

  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto is_reorder_dw_dx_in_previous_pass =
    (ms_context->get_param<bool>(MS_CTX_GRAD_COMM_OVERLAP) ||
     parallel::ParallelContext::GetInstance()->enable_fine_grained_micro_interleaved());
  if (is_reorder_dw_dx_in_previous_pass) {
    // Insert depend according grad_reduce in order
    auto pre_grad_reduce_user = grad_reduce_user_list.front();
    for (size_t i = 1; i < grad_reduce_user_list.size(); ++i) {
      auto cur_grad_reduce_user = grad_reduce_user_list.at(i);
      auto pre_grad_reduce_cnode = pre_grad_reduce_user.grad_reduce_list.back();
      auto cur_grad_reduce_cnode = cur_grad_reduce_user.grad_reduce_list.front();
      InsertDepend(pre_grad_reduce_cnode, cur_grad_reduce_cnode, manager, kernel_graph, "grad_comm_in_order_depend");
      pre_grad_reduce_user = cur_grad_reduce_user;
    }
    // Insert depend between grad reduce and next op
    for (size_t i = 0; i < grad_reduce_user_list.size() - 1; ++i) {
      auto cur_grad_reduce_user = grad_reduce_user_list.at(i);
      auto next_grad_reduce_user = grad_reduce_user_list.at(i + 1);
      auto cur_grad_reduce_cnode = cur_grad_reduce_user.grad_reduce_list.back();

      auto cur_grad_compute_node = cur_grad_reduce_user.latest_dw_compute_node;
      auto next_grad_compute_node = next_grad_reduce_user.latest_dw_compute_node;
      InsertDepend(cur_grad_compute_node, next_grad_compute_node, manager, kernel_graph, "dw_in_order_depend");
      InsertDepend(cur_grad_reduce_cnode, next_grad_compute_node, manager, kernel_graph, "grad_comm_next_dw_depend");
    }
  } else {
    // Insert depend node to correspond dx
    for (size_t i = 0; i < grad_reduce_user_list.size(); ++i) {
      auto cur_grad_reduce_user = grad_reduce_user_list.at(i);
      auto cur_grad_reduce_cnode = cur_grad_reduce_user.grad_reduce_list.back();
      auto cur_grad_compute_node = cur_grad_reduce_user.latest_dw_compute_node;

      if (!IsOneOfPrimitiveCNode(cur_grad_compute_node, {prim::kPrimMatMul, prim::kPrimBatchMatMul,
                                                         prim::kPrimMatMulExt, prim::kPrimBatchMatMulExt})) {
        continue;
      }
      // find corresponding dx computation
      auto dx_compute_node = FindDxMatMulByDw(cur_grad_compute_node);
      MS_EXCEPTION_IF_NULL(dx_compute_node);
      InsertDepend(cur_grad_reduce_cnode, dx_compute_node, manager, kernel_graph, "grad_comm_dx_depend");
    }
  }
  return true;
}

bool OverlapGradReduce::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph == nullptr) {
    MS_LOG(DEBUG) << "Failed convert func_graph to kernel_graph, skip it.";
    return false;
  }
  MS_LOG(DEBUG) << "Status record: start overlap grad reduce optimization. graph id: " << kernel_graph->graph_id();
  auto parallel_context = parallel::ParallelContext::GetInstance();
  MS_EXCEPTION_IF_NULL(parallel_context);

  auto stages = parallel_context->pipeline_stage_split_num();
  auto grad_accu_step = parallel_context->grad_accumulation_step();
  auto with_accumulation = stages > 1 || grad_accu_step > 1;
  auto stage_device_num = parallel_context->device_num() / stages;
  auto stage_id = parallel_context->global_rank() / stage_device_num;
  bool ret;
  if (stage_id > 0) {
    MS_LOG(DEBUG) << "Under pipeline parallelism, stage " << stage_id
                  << " no need to reorder grad reduce communication, skip it.";
    ret = false;
  } else {
    ret = DoOverlapGradReduce(kernel_graph, with_accumulation);
  }
  MS_LOG(DEBUG) << "Status record: end overlap grad reduce optimization. graph id: " << kernel_graph->graph_id();
  return ret;
}
}  // namespace opt
}  // namespace mindspore
