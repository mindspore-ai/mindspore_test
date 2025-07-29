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
#include "frontend/optimizer/irpass/insert_remote_ops_before_grad.h"

#include <vector>
#include "pipeline/jit/ps/remote_memory.h"

namespace mindspore {
namespace opt {
namespace irpass {
namespace {
std::vector<FuncGraphPtr> CollectForwardGraphs(const FuncGraphPtr &func_graph) {
  const AnfNodePtrList &nodes = mindspore::TopoSort(func_graph->get_return(), SuccDeeperSimple);
  std::vector<FuncGraphPtr> ret;
  for (auto node : nodes) {
    if (!IsPrimitiveCNode(node, prim::kPrimJ)) {
      continue;
    }
    auto cnode = node->cast<CNodePtr>();
    constexpr size_t forward_index = 1;
    auto forward_fg = GetValueNode<FuncGraphPtr>(cnode->input(forward_index));
    MS_EXCEPTION_IF_NULL(forward_fg);
    (void)ret.emplace_back(forward_fg);
  }
  return ret;
}
}  // namespace

bool InsertRemoteOpsBeforeGrad::operator()(const FuncGraphPtr &func_graph, const OptimizerPtr &optimizer) {
  const auto &forward_graphs = CollectForwardGraphs(func_graph);
  if (forward_graphs.empty()) {
    MS_LOG(ERROR) << "No forward graph for grad, no need to insert remote ops.";
    return false;
  }
  MS_LOG(ERROR) << "Start to insert remote ops for forward graph";
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);

  bool change = false;
  for (auto forward_graph : forward_graphs) {
    auto cur_change = remote_memory::InsertActivactionRemoteOpsForGraph(mng, func_graph);
    change = change || cur_change;
  }
  return change;
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
