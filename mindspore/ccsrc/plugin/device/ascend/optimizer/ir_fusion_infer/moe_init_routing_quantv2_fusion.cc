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
#include "plugin/device/ascend/optimizer/ir_fusion_infer/moe_init_routing_quantv2_fusion.h"

#include <string>
#include <vector>
#include <set>
#include <functional>

#include "backend/common/pass/common/gllo_utils.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/inference_weight_preprocess_utils.h"
#include "ir/primitive.h"
#include "utils/shape_utils.h"
#include "utils/ms_context.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_q.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace opt {
std::vector<std::string> MoeInitRoutingQuantV2Fusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimMoeInitRoutingV2->name(), prim::kPrimQuantV2->name()};
  return ret;
}

const BaseRef MoeInitRoutingQuantV2Fusion::DefinePattern() const {
  auto moe_int_routing =
    VectorRef({prim::kPrimMoeInitRoutingV2, x_, expert_idx_, active_num_, expert_capacity_, expert_num_, drop_pad_mode_,
               expert_tokens_count_or_cumsum_flag_, expert_tokens_before_capacity_flag_});
  VarPtr index_0 = std::make_shared<CondVar>(IsConstant);
  auto tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, moe_int_routing, index_0});
  auto rounding_mode = std::make_shared<CondVar>(IsConstant);
  auto dst_type = std::make_shared<CondVar>(IsConstant);
  auto sqrt_mode = std::make_shared<CondVar>(IsConstant);
  auto quant_v2 =
    VectorRef({prim::kPrimQuantV2, tuple_get_item_0, scale_, offset_, sqrt_mode, rounding_mode, dst_type});
  return quant_v2;
}

static void ReplaceMoeInitRoutingV2Out(const FuncGraphPtr &graph, const FuncGraphManagerPtr &mng,
                                       const AnfNodePtr &moe_node, const int64_t replace_idx,
                                       const AnfNodePtr &replace_out) {
  auto moe_init_routing_out_list = GetRealNodeUsedList(graph, moe_node);
  for (const auto &out_user : *moe_init_routing_out_list) {
    auto &ori_getitem = out_user.first;
    auto item_index = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(ori_getitem), 1);
    auto item_index_ptr = item_index->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(item_index_ptr);
    auto idx = GetValue<int64_t>(item_index_ptr->value());
    if (idx == replace_idx) {
      (void)mng->Replace(ori_getitem, replace_out);
    }
  }
}

static AnfNodePtr NewGetIteamOut(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                 const AnfNodePtr &moe_init_routing_quantv2, const int32_t index,
                                 const TypeId &ori_type, const BaseShapePtr &ori_shape) {
  auto getitem_prim = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> get_item_node = {NewValueNode(getitem_prim), moe_init_routing_quantv2,
                                           NewValueNode(MakeValue((int64_t)index))};
  std::vector<TypeId> get_item_types{ori_type};
  std::vector<BaseShapePtr> get_item_shapes{ori_shape};
  auto item_cnode = graph->NewCNode(get_item_node);
  common::AnfAlgo::SetOutputTypeAndDetailShape(get_item_types, get_item_shapes, item_cnode.get());
  item_cnode->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(item_cnode);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, item_cnode.get());
  return item_cnode;
}

bool MoeInitRoutingQuantV2Fusion::IsSupport(const AnfNodePtr &node, const EquivPtr &equiv) const {
  auto x = utils::cast<AnfNodePtr>((*equiv)[x_]);
  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32};
  if (!CheckSupportDataType(x, support_dtype)) {
    MS_LOG(INFO) << "MoeInitRoutingDynQuantV2Fusion only support type float16, bfloat16, float32";
    return false;
  }

  auto drop_pad_mode = utils::cast<AnfNodePtr>((*equiv)[drop_pad_mode_]);
  auto drop_pad_mode_cnode = drop_pad_mode->cast<ValueNodePtr>();
  if (drop_pad_mode_cnode == nullptr) {
    MS_LOG(INFO) << "drop_pad_mode_cnode is not a value node";
    return false;
  }

  auto drop_pad_mode_value = GetValue<int64_t>(drop_pad_mode_cnode->value());
  if (drop_pad_mode_value == 1) {
    MS_LOG(INFO) << "MoeInitRoutingDynQuantV2Fusion do not support Drop/Pad Mode";
    return false;
  }

  auto scale_cnode = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  if (scale_cnode == nullptr) {
    MS_LOG(INFO) << "scale_cnode is not a value node";
    return false;
  }

  auto offset_cnode = utils::cast<AnfNodePtr>((*equiv)[offset_]);
  if (offset_cnode == nullptr) {
    MS_LOG(INFO) << "offset_cnode is not a value node";
    return false;
  }

  auto smooth_scale_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, kScaleOutIdx);
  if (smooth_scale_shape.size() != 1) {
    MS_LOG(INFO) << "do not support fusion when smooth_scale shape isn't a 1D tensor, smooth_scale_shape = "
                 << smooth_scale_shape;
    return false;
  }

  auto offset_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(node, kOffsetOutIdx);
  if (offset_shape.size() != 1) {
    MS_LOG(INFO) << "do not support fusion when smooth_scale shape isn't a 1D tensor, offset_shape = " << offset_shape;
    return false;
  }

  return true;
}

CNodePtr MoeInitRoutingQuantV2Fusion::CreateMoeInitRoutingQuantV2Node(const FuncGraphPtr &func_graph,
                                                                      const AnfNodePtr &node,
                                                                      const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "start create MoeInitRoutingQuantV2 node";
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  FuncGraphManagerPtr mng = func_graph->manager();
  MS_EXCEPTION_IF_NULL(mng);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto constexpr kExpandx = 0;
  auto constexpr kExpandRowIdx = 1;
  auto constexpr kCumsumOutIdx = 2;
  auto constexpr kCapacityOutIdx = 3;

  auto x = utils::cast<AnfNodePtr>((*equiv)[x_]);
  auto expert_idx = utils::cast<AnfNodePtr>((*equiv)[expert_idx_]);
  auto scale = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  auto offset = utils::cast<AnfNodePtr>((*equiv)[offset_]);
  auto active_num = utils::cast<AnfNodePtr>((*equiv)[active_num_]);
  auto expert_capacity = utils::cast<AnfNodePtr>((*equiv)[expert_capacity_]);
  auto expert_num = utils::cast<AnfNodePtr>((*equiv)[expert_num_]);
  auto drop_pad_mode = utils::cast<AnfNodePtr>((*equiv)[drop_pad_mode_]);
  auto expert_tokens_count_or_cumsum_flag = utils::cast<AnfNodePtr>((*equiv)[expert_tokens_count_or_cumsum_flag_]);
  auto expert_tokens_before_capacity_flag = utils::cast<AnfNodePtr>((*equiv)[expert_tokens_before_capacity_flag_]);

  if (!IsSupport(node, equiv)) return nullptr;

  auto tuple_getitem_0 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto moe_init_routing_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tuple_getitem_0), 0);
  auto expand_x_out_type = common::AnfAlgo::GetOutputInferDataType(moe_init_routing_node, 0);

  AnfNodePtr scale_node = scale;
  AnfNodePtr offset_node = offset;
  if (expand_x_out_type != kNumberTypeFloat32) {
    auto scale_fp32 = ConvertWeightsToNewType(scale);
    auto offset_fp32 = ConvertWeightsToNewType(offset);
    kernel_graph->AddValueNodeToGraph(scale_fp32);
    kernel_graph->AddValueNodeToGraph(offset_fp32);
    scale_node = utils::cast<AnfNodePtr>(scale_fp32);
    offset_node = utils::cast<AnfNodePtr>(offset_fp32);
  }
  auto quant_mode = kernel_graph->NewValueNode(MakeValue((int64_t)(0)));
  kernel_graph->AddValueNodeToGraph(quant_mode);

  auto moe_init_routing_quantv2_prim = std::make_shared<Primitive>("MoeInitRoutingQuantV2");
  std::vector<AnfNodePtr> quant_inputs = {x,
                                          expert_idx,
                                          active_num,
                                          expert_capacity,
                                          expert_num,
                                          drop_pad_mode,
                                          expert_tokens_count_or_cumsum_flag,
                                          expert_tokens_before_capacity_flag,
                                          quant_mode,
                                          scale_node,
                                          offset_node};
  auto moe_init_routing_quantv2_node = func_graph->NewCNode(moe_init_routing_quantv2_prim, quant_inputs);
  MS_EXCEPTION_IF_NULL(moe_init_routing_quantv2_node);

  std::vector<TypeId> moe_init_routing_quantv2_types;
  std::vector<BaseShapePtr> moe_init_routing_quantv2_shapes;
  auto quant_out_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  auto quant_out_shape = AnfAlgo::GetOutputDetailShape(node, 0);
  auto expanded_row_idx_type = common::AnfAlgo::GetOutputInferDataType(moe_init_routing_node, kExpandRowIdx);
  auto expanded_row_idx_shape = AnfAlgo::GetOutputDetailShape(moe_init_routing_node, kExpandRowIdx);
  auto expert_tokens_count_or_cumsum_type =
    common::AnfAlgo::GetOutputInferDataType(moe_init_routing_node, kCumsumOutIdx);
  auto expert_tokens_count_or_cumsum_shape = AnfAlgo::GetOutputDetailShape(moe_init_routing_node, kCumsumOutIdx);
  auto expert_tokens_before_capacity_type =
    common::AnfAlgo::GetOutputInferDataType(moe_init_routing_node, kCapacityOutIdx);
  auto expert_tokens_before_capacity_shape = AnfAlgo::GetOutputDetailShape(moe_init_routing_node, kCapacityOutIdx);

  auto shape_vec = quant_out_shape->GetShapeVector();
  auto input_element =
    std::accumulate(shape_vec.begin(), shape_vec.end() - 1, static_cast<int64_t>(1), std::multiplies<int64_t>());
  ShapeVector out_scale_shape_vec;
  out_scale_shape_vec.push_back(input_element);
  auto out_scale_shape = std::make_shared<abstract::Shape>(out_scale_shape_vec);
  // auto out_scale_shape = std::make_shared<abstract::NoShape>();

  moe_init_routing_quantv2_types.push_back(quant_out_type);
  moe_init_routing_quantv2_shapes.push_back(quant_out_shape);
  moe_init_routing_quantv2_types.push_back(expanded_row_idx_type);
  moe_init_routing_quantv2_shapes.push_back(expanded_row_idx_shape);
  moe_init_routing_quantv2_types.push_back(expert_tokens_count_or_cumsum_type);
  moe_init_routing_quantv2_shapes.push_back(expert_tokens_count_or_cumsum_shape);
  moe_init_routing_quantv2_types.push_back(expert_tokens_before_capacity_type);
  moe_init_routing_quantv2_shapes.push_back(expert_tokens_before_capacity_shape);
  moe_init_routing_quantv2_types.push_back(TypeId::kNumberTypeFloat32);
  moe_init_routing_quantv2_shapes.push_back(out_scale_shape);

  common::AnfAlgo::SetOutputTypeAndDetailShape(moe_init_routing_quantv2_types, moe_init_routing_quantv2_shapes,
                                               moe_init_routing_quantv2_node.get());
  moe_init_routing_quantv2_node->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(moe_init_routing_quantv2_node);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, moe_init_routing_quantv2_node.get());

  auto y_out =
    NewGetIteamOut(func_graph, node, moe_init_routing_quantv2_node, kExpandx, quant_out_type, quant_out_shape);
  auto output_node = y_out->cast<CNodePtr>();
  auto expand_idx_out = NewGetIteamOut(func_graph, node, moe_init_routing_quantv2_node, kExpandRowIdx,
                                       expanded_row_idx_type, expanded_row_idx_shape);
  auto cumsum_out = NewGetIteamOut(func_ + graph, node, moe_init_routing_quantv2_node, kCumsumOutIdx,
                                   expert_tokens_count_or_cumsum_type, expert_tokens_count_or_cumsum_shape);
  auto capacity_out = NewGetIteamOut(func_graph, node, moe_init_routing_quantv2_node, kCapacityOutIdx,
                                     expert_tokens_before_capacity_type, expert_tokens_before_capacity_shape);
  ReplaceMoeInitRoutingV2Out(func_graph, mng, moe_init_routing_node, kExpandRowIdx, expand_idx_out);
  ReplaceMoeInitRoutingV2Out(func_graph, mng, moe_init_routing_node, kCumsumOutIdx, cumsum_out);
  ReplaceMoeInitRoutingV2Out(func_graph, mng, moe_init_routing_node, kCapacityOutIdx, capacity_out);
  return output_node;
}

const AnfNodePtr MoeInitRoutingQuantV2Fusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                      const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  auto soc = ms_context->ascend_soc_version();
  if (!soc.empty() && soc.find("ascend910_93") == std::string::npos && soc.find("ascend910b") == std::string::npos) {
    MS_LOG(INFO) << "MoeInitRoutingQuantV2Fusion does not support " << soc;
    return nullptr;
  }

  auto cnode = CreateMoeInitRoutingQuantV2Node(graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "create MoeInitRoutingQuantV2Fusion node failed.";
    return nullptr;
  }
  MS_LOG(DEBUG) << "create moe_init_routing_quantv2 node success.";
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
