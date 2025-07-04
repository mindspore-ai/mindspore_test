/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "frontend/optimizer/irpass/meta_fg_prim_eliminate.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore {
namespace opt {
namespace irpass {
bool ExpandMetaFgPrim::CheckIfEmbedMetaFgPrim(const CNodePtr &node) const {
  auto &value_node = node->input(1);
  if (IsValueNode<Primitive>(value_node)) {
    return false;
  }
  auto func_graph = GetValueNode<FuncGraphPtr>(value_node);
  if (func_graph == nullptr) {
    MS_LOG_WITH_NODE(INTERNAL_EXCEPTION, node) << "Unexpected meta function graph node:" << node->DebugString();
  }
  auto func_graph_manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(func_graph_manager);
  return func_graph_manager->func_graph_meta_fg_prim_total(func_graph);
}

void ExpandMetaFgPrim::GetMetaFgPrim(const AnfNodeSet &all_nodes) {
  MS_EXCEPTION_IF_NULL(prim_);
  prim_nodes_.clear();
  for (auto &node : all_nodes) {
    auto add_prim = false;
    auto execution_mode = MsContext::GetInstance()->get_param<int>(MS_CTX_EXECUTION_MODE);
    if ((execution_mode == kGraphMode) && IsPrimitiveCNode(node, prim::kPrimShard)) {
      add_prim = true;
    }
    if (IsPrimitiveCNode(node, prim_) && !CheckIfEmbedMetaFgPrim(node->cast<CNodePtr>())) {
      add_prim = true;
    }
    if (add_prim) {
      prim_nodes_.push_back(node->cast<CNodePtr>());
    }
  }
}
}  // namespace irpass
}  // namespace opt
}  // namespace mindspore
