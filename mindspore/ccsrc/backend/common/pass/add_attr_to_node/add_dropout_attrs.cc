/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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

#include "backend/common/pass/add_attr_to_node/add_attr_to_node_register.h"

#include <memory>
#include "include/backend/optimizer/helper.h"
#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "include/common/utils/anfalgo.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
const AnfNodePtr AddDropoutAttrs(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  if (!IsPrimitiveCNode(node, prim::kPrimDropout)) {
    return nullptr;
  }

  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  bool only_use_first_output = true;
  bool only_use_second_output = true;
  bool no_user = true;
  for (const auto &node_user : manager->node_users()[cnode]) {
    no_user = false;
    MS_EXCEPTION_IF_NULL(node_user.first);
    auto user = node_user.first->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(user);
    if (!common::AnfAlgo::CheckPrimitiveType(user, prim::kPrimTupleGetItem)) {
      only_use_first_output = false;
      only_use_second_output = false;
      break;
    }
    int64_t used_output_index = GetGetitemIndex(user);
    if (used_output_index == 0) {
      only_use_second_output = false;
    } else if (used_output_index == 1) {
      only_use_first_output = false;
    }
  }
  if (no_user) {
    return nullptr;
  }
  if (only_use_first_output) {
    cnode->AddAttr(kAttrOnlyUseFirstOutput, MakeValue(true));
  }
  if (only_use_second_output) {
    cnode->AddAttr(kAttrOnlyUseSecondOutput, MakeValue(true));
  }
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
