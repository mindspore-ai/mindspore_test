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

#include "plugin/device/ascend/optimizer/ge/convert_data_depend_to_control_depend.h"
#include <vector>
#include <memory>
#include "mindspore/ops/op_def/framework_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/transform/graph_ir/utils.h"

namespace mindspore {
namespace opt {
namespace {
const auto kDataToControl = "data_to_control";
}

const AnfNodePtr ConvertDataDependToControlDepend::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                           const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  if (common::AnfAlgo::HasNodeAttr(kDataToControl, cnode) || !AnfUtils::IsRealKernel(node) ||
      IsPrimitiveCNode(node, prim::kPrimInitDataSetQueue)) {
    return nullptr;
  }
  auto adapter = transform::FindAdapter(node, True);
  if (adapter == nullptr || !adapter->getOutputMap().empty() || !adapter->getDynOutputMap().empty()) {
    return nullptr;
  }

  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  auto node_users = manager->node_users()[cnode];

  // Check if there exists a node that uses a node with no output as a data input.
  auto res = std::find_if(node_users.begin(), node_users.end(), [&cnode](const std::pair<AnfNodePtr, int> &user) {
    if (IsPrimitiveCNode(user.first, prim::kPrimUpdateState)) {
      return false;
    }
    auto user_cnode = user.first->cast<CNodePtr>();
    if (IsPrimitiveCNode(user.first, prim::kPrimDepend) && user_cnode->input(1) != cnode) {
      return false;
    }
    return true;
  });
  if (res == node_users.end()) {
    return nullptr;
  }

  MS_LOG(DEBUG) << "Process node: " << node->fullname_with_scope();
  auto tensor = std::make_shared<tensor::Tensor>(0.0);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);
  ValueNodePtr value_node = kernel_graph->NewValueNode(tensor->ToAbstract(), tensor);
  kernel_graph->AddValueNodeToGraph(value_node);

  std::vector<AnfNodePtr> depend_input = {NewValueNode(std::make_shared<Primitive>(kDependOpName)), value_node, cnode};
  auto depend_node = NewCNode(depend_input, func_graph);
  MS_EXCEPTION_IF_NULL(depend_node);
  depend_node->set_scope(node->scope());
  depend_node->set_abstract(value_node->abstract());
  MS_LOG(INFO) << "Create new depend: " << depend_node->fullname_with_scope()
               << " for node: " << cnode->fullname_with_scope();
  common::AnfAlgo::SetNodeAttr(kDataToControl, MakeValue(true), cnode);
  return depend_node;
}
}  // namespace opt
}  // namespace mindspore
