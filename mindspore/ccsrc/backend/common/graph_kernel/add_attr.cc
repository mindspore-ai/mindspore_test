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
#include "backend/common/graph_kernel/add_attr.h"

#include <set>
#include <string>

#include "include/common/utils/anfalgo.h"
#include "mindspore/ops/op_def/other_ops.h"  // collective communication operations

namespace mindspore::graphkernel {
bool AddAttr::Process(const AnfNodePtr &graph_kernel_node) const {
  bool changed{false};
  static const std::set<std::string> kCommOpsNames = {kAllReduceOpName, kAllGatherOpName, kReduceScatterOpName};
  auto sub_graph = GetCNodeFuncGraph(graph_kernel_node);
  MS_EXCEPTION_IF_NULL(sub_graph);
  auto nodes = TopoSort(sub_graph->get_return());
  for (auto node : nodes) {
    auto prim = GetCNodePrimitive(node);
    if (prim != nullptr && kCommOpsNames.find(prim->name()) != kCommOpsNames.end()) {
      changed = true;
      common::AnfAlgo::SetNodeAttrSafely("is_comm_op", MakeValue(true), graph_kernel_node);
    }
  }
  return changed;
}

bool AddAttr::Run(const FuncGraphPtr &func_graph) {
  MS_EXCEPTION_IF_NULL(func_graph);
  auto mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  bool changed{false};
  auto todos = TopoSort(func_graph->get_return());
  for (auto node : todos) {
    if (common::AnfAlgo::IsGraphKernel(node)) {
      changed = Process(node);
    }
  }
  if (changed) {
    mng->RemoveRoots();
    mng->KeepRoots({func_graph});
  }
  return changed;
}
}  // namespace mindspore::graphkernel
