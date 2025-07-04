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
#include "backend/common/pass/add_attr_to_node/add_attr_to_node_register.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/kernel_graph.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "utils/anf_utils.h"

namespace mindspore {
namespace opt {
const AnfNodePtr SqueezeAxis(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  auto squeeze_cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(squeeze_cnode);
  auto prim = common::AnfAlgo::GetCNodePrimitive(squeeze_cnode);
  MS_EXCEPTION_IF_NULL(prim);
  auto axis_index = squeeze_cnode->size() - 1;
  auto axis_value = squeeze_cnode->input(axis_index);
  MS_EXCEPTION_IF_NULL(axis_value);
  if (!axis_value->isa<ValueNode>()) {
    MS_LOG(WARNING) << "Squeeze node axis can not support dynamic, when axis is empty tuple";
    return node;
  }
  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(squeeze_cnode, kIndex0);
  if (IsDynamicRank(x_shape)) {
    MS_LOG(WARNING) << "For '" << squeeze_cnode->fullname_with_scope()
                    << "', x_shape can not support dynamic rank, when axis is empty tuple.";
    return node;
  }
  auto value_node = axis_value->cast<ValueNodePtr>();
  auto actual_value = value_node->value();
  MS_EXCEPTION_IF_CHECK_FAIL(actual_value->isa<ValueSequence>(),
                             "Squeeze node axis attr error, squeeze node: " + squeeze_cnode->DebugString() +
                               ", axis value: " + axis_value->ToString());
  auto &value_sequence = actual_value->cast<ValueSequencePtr>()->value();
  auto shape_vec = common::AnfAlgo::GetOutputInferShape(squeeze_cnode->input(kIndex1), 0);
  const auto dim = shape_vec.size();
  auto kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(graph);
  MS_EXCEPTION_IF_NULL(kernel_graph);
  std::vector<int64_t> axis;
  if (value_sequence.empty()) {
    for (size_t i = 0; i < dim; ++i) {
      if (shape_vec[i] != 1) {
        continue;
      }
      (void)axis.emplace_back(i);
    }
    auto new_node = kernel_graph->NewValueNode(MakeValue(axis));
    squeeze_cnode->set_input(kIndex2, new_node);
    return node;
  }

  for (const auto &value : value_sequence) {
    auto axis_data = AnfUtils::GetIntValue(value);
    auto real_idx = (axis_data < 0) ? axis_data + SizeToLong(dim) : axis_data;
    (void)axis.emplace_back(real_idx);
  }
  auto new_node = kernel_graph->NewValueNode(MakeValue(axis));
  squeeze_cnode->set_input(kIndex2, new_node);
  return node;
}
}  // namespace opt
}  // namespace mindspore
