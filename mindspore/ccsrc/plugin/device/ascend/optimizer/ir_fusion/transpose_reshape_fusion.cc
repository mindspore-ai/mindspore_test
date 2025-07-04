/**
 * Copyright 2020-2023 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/optimizer/ir_fusion/transpose_reshape_fusion.h"
#include <memory>
#include "mindspore/ops/op_def/ascend_op_name.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/optimizer/ir_fusion/reshape_transpose_fusion.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"

namespace mindspore {
namespace opt {
const BaseRef TransposeReshapeFusion::DefinePattern() const {
  const auto prim_reshape = std::make_shared<Primitive>(prim::kPrimReshape->name());
  VectorRef transpose({prim::kPrimTransposeD, input_varptr_});

  return VectorRef({prim_reshape, transpose});
}

const AnfNodePtr TransposeReshapeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(equiv);

  auto reshape_cnode = CheckAnfNodeIfCNodeAndInputSize(node, kBackendReshapeInputTensorNum);
  MS_EXCEPTION_IF_NULL(reshape_cnode);
  auto transpose_cnode = CheckAnfNodeIfCNodeAndInputSize(reshape_cnode->input(1), kBackendReshapeInputTensorNum);
  MS_EXCEPTION_IF_NULL(transpose_cnode);
  if (common::AnfAlgo::IsDynamicShape(transpose_cnode) || common::AnfAlgo::IsDynamicShape(reshape_cnode)) {
    return nullptr;
  }
  if (!CheckMatmulNeighborNodes(func_graph, transpose_cnode, reshape_cnode)) {
    return nullptr;
  }
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (kernel_graph != nullptr &&
      (kernel_graph->IsInternalOutput(reshape_cnode, 0) || kernel_graph->IsInternalOutput(transpose_cnode, 0))) {
    return nullptr;
  }
  auto reshape_output0_shape = common::AnfAlgo::GetOutputInferShape(reshape_cnode, 0);
  auto transpose_input0_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(transpose_cnode, 0);
  if (!CheckShapeDimInfo(reshape_output0_shape) || !CheckShapeDimInfo(transpose_input0_shape)) {
    return nullptr;
  }
  auto prim = std::make_shared<Primitive>(kConfusionTransposeDOpName);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), utils::cast<AnfNodePtr>((*equiv)[input_varptr_])};
  auto new_node = NewCNode(inputs, func_graph);
  MS_EXCEPTION_IF_NULL(new_node);

  new_node->set_abstract(node->abstract());
  common::AnfAlgo::CopyNodeAttrs(reshape_cnode, new_node);
  common::AnfAlgo::CopyNodeAttr(kAttrPerm, transpose_cnode, new_node);
  common::AnfAlgo::SetNodeAttr(kAttrTransposeFirst, MakeValue(true), new_node);
  auto reshape_output_shape = common::AnfAlgo::GetOutputInferShape(reshape_cnode, 0);
  common::AnfAlgo::SetNodeAttr(kAttrShape, MakeValue(reshape_output_shape), new_node);

  return new_node;
}
}  // namespace opt
}  // namespace mindspore
