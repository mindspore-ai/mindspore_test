/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "backend/ge_backend/pass/expander_fallback.h"
#include <vector>
#include "backend/common/expander/fallback/expander_fallback.h"
#include "backend/common/pass/value_graph_binder.h"
#include "common/device_type.h"
#include "include/backend/kernel_graph.h"
#include "include/common/utils/anfalgo.h"
#include "kernel/ascend/opapi/aclnn_kernel_build.h"
#include "plugin/res_manager/ascend/op_adapter/op_adapter_util.h"

namespace mindspore {
namespace opt {
bool ExpanderFallback::Run(const FuncGraphPtr &graph) {
  auto kernel_graph = graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  auto IsRegisteredAdapter = [](const AnfNodePtr &node) { return device::ascend::ConvertCheck(node); };

  bool changed = false;
  const std::vector<AnfNodePtr> node_list = TopoSort(graph->get_return());
  for (auto &node : node_list) {
    MS_EXCEPTION_IF_NULL(node);
    if (!IsRegisteredAdapter(node)) {
      auto f = [](const CNodePtr &n) { return true; };
      changed = expander::TryExpandCNode(node, f) || changed;
    }
  }
  if (changed) {
    auto all_nodes = TopoSort(graph->get_return());
    for (const auto &node : all_nodes) {
      if (common::AnfAlgo::IsDynamicShape(node)) {
        kernel_graph->SetGraphDynamicAttr(true);
        break;
      }
    }
    BindValueToGraph().Run(graph);
  }
  return changed;
}
}  // namespace opt
}  // namespace mindspore
