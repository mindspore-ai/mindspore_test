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
#include "backend/common/pass/ir_fission/ascend_convert_tuple_input_to_dynamic_input.h"
#include <memory>
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/structure_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "mindspore/ops/op_def/framework_ops.h"
#include "include/backend/optimizer/helper.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_f.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_i.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_p.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
const BaseRef AscendConvertTupleInputToDynamicInput::DefinePattern() const {
  VarPtr V = std::make_shared<Var>();
  VarPtr Xs = std::make_shared<SeqVar>();
  return VectorRef({V, Xs});
}

const AnfNodePtr AscendConvertTupleInputToDynamicInput::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                                const EquivPtr &) const {
  if (node == nullptr || !node->isa<CNode>() || !AnfUtils::IsRealKernel(node)) {
    return nullptr;
  }
  auto cnode = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(cnode);
  // since the input should be unfolded before some function, this pass should be in front of concat_fission,
  // pack_fission, addn_fission, and HandleControlFlow

  static const PrimitiveSet need_unfold_calculate_node = {prim::kPrimAddN,
                                                          prim::kPrimConcatD,
                                                          prim::kPrimPack,
                                                          prim::kPrimStack,
                                                          prim::kPrimPrint,
                                                          prim::kPrimConcat,
                                                          prim::kPrimAccumulateNV2,
                                                          prim::kPrimMeshgrid,
                                                          prim::kPrimTensorSummary,
                                                          prim::kPrimDynamicStitch,
                                                          prim::kPrimParallelConcat,
                                                          prim::kPrimIncreFlashAttention,
                                                          prim::kPrimIdentityN,
                                                          prim::kPrimConcatOffset,
                                                          prim::kPrimAllFinite,
                                                          prim::kPrimFusedInferAttentionScore,
                                                          prim::kPrimGroupedMatmul,
                                                          prim::kPrimStackExt,
                                                          prim::kPrimInnerIndex,
                                                          prim::kPrimInnerInplaceIndexPut,
                                                          prim::kPrimGroupedMatmulV4,
                                                          prim::kPrimGroupedMatmulV2,
                                                          prim::kPrimCustom};

  static const PrimitiveSet need_unfold_control_node = {prim::kPrimSwitchLayer, prim::kPrimCall, prim::kPrimSwitch,
                                                        prim::kPrimCallInline};
  PrimitivePtr prim = common::AnfAlgo::GetCNodePrimitive(cnode);
  MS_EXCEPTION_IF_NULL(prim);
  bool is_communication_op = common::AnfAlgo::IsCommunicationOp(node);
  bool not_unfold_communication_op = false;
  if (common::AnfAlgo::GetCNodeName(node) == kBroadcastOpName) {
    not_unfold_communication_op = true;
  }
  bool is_unfold_calculate_op = IsOneOfPrimitiveCNode(node, need_unfold_calculate_node);
  bool is_unfold_control_op = IsOneOfPrimitiveCNode(node, need_unfold_control_node);
  // In GE backend, control node should not be unfold.
  if (is_ge_ && is_unfold_calculate_op) {
    return ConvertMakeTupleInputToPlantInputs(func_graph, cnode);
  } else if (!is_ge_ && (is_communication_op || is_unfold_calculate_op || is_unfold_control_op) &&
             !not_unfold_communication_op) {
    return ConvertMakeTupleInputToPlantInputs(func_graph, cnode);
  }
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
