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

#include "plugin/device/ascend/optimizer/ir_fusion/insert_depend_for_all_gather.h"
#include <unordered_map>
#include <queue>
#include <vector>
#include <utility>
#include "mindspore/ops/op_def/framework_ops.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/other_ops.h"
#include "include/common/utils/utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/parallel_context.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace opt {
namespace {
AnfNodePtr InputNodeWithFilter(const AnfNodePtr &node,
                               std::function<std::pair<bool, size_t>(const CNodePtr &)> filter) {
  std::queue<AnfNodePtr> anf_node_queue;
  anf_node_queue.push(node);
  while (!anf_node_queue.empty()) {
    auto node_queue_end = anf_node_queue.front();
    anf_node_queue.pop();
    if (!node_queue_end->isa<CNode>()) {
      return node_queue_end;
    }
    auto cnode_queue_end = node_queue_end->cast<CNodePtr>();
    auto filter_res = filter(cnode_queue_end);
    if (!filter_res.first) {
      return node_queue_end;
    }
    anf_node_queue.push(cnode_queue_end->input(filter_res.second));
  }
  return node;
}

CNodePtr CreateDepend(const AnfNodePtr &first_input, const AnfNodePtr &second_input, const FuncGraphPtr &graph) {
  std::vector<AnfNodePtr> inputs = {NewValueNode(std::make_shared<Primitive>(prim::kPrimDepend->name())), first_input,
                                    second_input};
  auto new_input = graph->NewCNode(inputs);
  new_input->set_abstract(first_input->abstract());
  new_input->set_scope(first_input->scope());
  return new_input;
}

bool InsertDepend(const FuncGraphPtr &graph, const std::vector<CNodePtr> &allgather_with_output_order,
                  const std::unordered_map<CNodePtr, CNodePtr> &allgather_output_another_input,
                  const std::unordered_map<CNodePtr, CNodePtr> &allgather_output) {
  bool changed = false;
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  for (size_t i = 0; i + 1 < allgather_with_output_order.size(); ++i) {
    auto current_ag_node = allgather_with_output_order[i];
    auto next_ag_node = allgather_with_output_order[i + 1];
    // current_ag_node -> next_ag_node
    auto depend1 = CreateDepend(next_ag_node->input(1), current_ag_node, graph);
    manager->SetEdge(next_ag_node, 1, depend1);
    depend1->AddAttr("opt_shard_depend1", MakeValue(true));
    changed = true;
    // allgather_output_another_input -> allgather
    if (allgather_output_another_input.count(current_ag_node) == 0) {
      continue;
    }

    // allgather_output_another_input -> next_ag
    auto depend2 = CreateDepend(next_ag_node->input(1), allgather_output_another_input.at(current_ag_node), graph);
    depend2->AddAttr("opt_shard_depend2", MakeValue(true));
    manager->SetEdge(next_ag_node, 1, depend2);

    // next_ag->current_output
    auto depend3 = CreateDepend(allgather_output.at(current_ag_node)->input(1), next_ag_node, graph);
    depend3->AddAttr("opt_shard_depend3", MakeValue(true));
    manager->SetEdge(allgather_output.at(current_ag_node), 1, depend3);
  }
  return changed;
}

bool IsValidAllGaterUser(const AnfNodePtr &node) {
  if (!node->cast<CNodePtr>() || !AnfUtils::IsRealKernel(node) || IsPrimitiveCNode(node, prim::kPrimCast)) {
    return false;
  }
  auto cnode = node->cast<CNodePtr>();
  bool is_recompute = cnode->GetAttr(kAttrDuplicated) != nullptr && GetValue<bool>(cnode->GetAttr(kAttrDuplicated));
  if (is_recompute) {
    return false;
  }
  return true;
}
}  // namespace

bool InsertDependForOptShardAllGather::Run(const FuncGraphPtr &graph) {
  MS_EXCEPTION_IF_NULL(graph);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (parallel::ParallelContext::GetInstance()->pipeline_stage_split_num() > 1) {
    MS_LOG(INFO) << "AllGather parallel optimization is not required in pipeline parallel mode.";
    return false;
  }
  const auto cell_reuse = ms_context->CellReuseLevel() != CellReuseLevel::kNoCellReuse;
  if (cell_reuse || AnfAlgo::GetBackend(graph) == kBackendGE) {
    return false;
  }

  std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  std::vector<CNodePtr> allgather_with_output_order;
  std::unordered_map<CNodePtr, CNodePtr> allgather_output_another_input;
  std::unordered_map<CNodePtr, CNodePtr> allgather_output;
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsValidAllGaterUser(node)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    size_t ag_index = 0;
    CNodePtr ag_node = nullptr;
    for (size_t i = 1; i < cnode->size(); ++i) {
      auto pre_cnode = InputNodeWithFilter(cnode->input(i), [&](const CNodePtr &cnode) {
        bool filter = IsPrimitiveCNode(cnode, prim::kPrimCast) || IsPrimitiveCNode(cnode, prim::kPrimLoad) ||
                      IsPrimitiveCNode(cnode, prim::kPrimDepend);
        return std::make_pair(filter, 1);
      });
      if (!IsPrimitiveCNode(pre_cnode, prim::kPrimAllGather) ||
          !common::AnfAlgo::IsFromParallelOptimizer(pre_cnode->cast<CNodePtr>())) {
        continue;
      }
      if (std::find(allgather_with_output_order.begin(), allgather_with_output_order.end(),
                    pre_cnode->cast<CNodePtr>()) != allgather_with_output_order.end()) {
        continue;
      }
      allgather_with_output_order.push_back(pre_cnode->cast<CNodePtr>());
      ag_index = i;
      ag_node = pre_cnode->cast<CNodePtr>();
      allgather_output[ag_node] = cnode;
    }
    for (size_t i = 1; i < cnode->size(); ++i) {
      if (ag_index > 0 && i != ag_index && ag_node && IsPrimitiveCNode(cnode->input(i))) {
        allgather_output_another_input[ag_node] = cnode->input(i)->cast<CNodePtr>();
      }
    }
  }

  bool changed = false;
  auto is_enable = ms_context->get_param<bool>(MS_CTX_ENABLE_OPT_SHARD_COMM_OPT);
  if (!is_enable) {
    return changed;
  }
  return InsertDepend(graph, allgather_with_output_order, allgather_output_another_input, allgather_output) || changed;
}
}  // namespace opt
}  // namespace mindspore
