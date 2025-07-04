/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/optimizer/matmul_biasadd_fusion.h"

#include <memory>
#include <vector>
#include <string>

#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_l.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace opt {
const BaseRef MatMulBiasAddFusion::DefinePattern() const {
  VectorRef load_w = VectorRef({prim::kPrimLoad, w_, u_});
  VectorRef load_bias = VectorRef({prim::kPrimLoad, bias_, u_});
  VectorRef bias_add = VectorRef({prim::kPrimBiasAdd, VectorRef({prim::kPrimMatMul, x_, load_w}), load_bias});
  return bias_add;
}

const AnfNodePtr MatMulBiasAddFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                              const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto u_input = utils::cast<AnfNodePtr>((*equiv)[u_]);
  auto x_input = utils::cast<AnfNodePtr>((*equiv)[x_]);
  auto w_input = utils::cast<AnfNodePtr>((*equiv)[w_]);
  auto bias_input = utils::cast<AnfNodePtr>((*equiv)[bias_]);
  MS_EXCEPTION_IF_NULL(u_input);
  MS_EXCEPTION_IF_NULL(x_input);
  MS_EXCEPTION_IF_NULL(w_input);
  MS_EXCEPTION_IF_NULL(bias_input);

  // The `Matmul` node should have an unique user.
  const AnfNodePtr &matmul = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(matmul);
  auto outlist = GetRealNodeUsedList(graph, matmul);
  MS_EXCEPTION_IF_NULL(outlist);
  if (outlist->size() >= 2) {
    return nullptr;
  }

  auto load_w = graph->NewCNode({NewValueNode(prim::kPrimLoad), w_input, u_input});
  MS_EXCEPTION_IF_NULL(load_w);
  load_w->set_abstract(w_input->abstract());
  auto load_bias = graph->NewCNode({NewValueNode(prim::kPrimLoad), bias_input, u_input});
  MS_EXCEPTION_IF_NULL(load_bias);
  load_bias->set_abstract(bias_input->abstract());

  // Fused into a FusedMatMulBiasAdd operator.
  auto prim = std::make_shared<Primitive>(kFusedMatMulBiasAddOpName);
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_value = NewValueNode(prim);
  std::vector<AnfNodePtr> inputs = {prim_value, x_input, load_w, load_bias};
  auto fused_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(fused_node);

  // Copy Abstract and KernelBuildInfo.
  auto types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(node, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, fused_node.get());
  common::AnfAlgo::CopyNodeAttrs(matmul, fused_node);
  fused_node->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(fused_node);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, fused_node.get());

  return fused_node;
}
}  // namespace opt
}  // namespace mindspore
