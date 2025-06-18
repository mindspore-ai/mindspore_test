
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
#include "plugin/device/ascend/optimizer/ir_fusion_infer/add_rms_norm_dynamic_quant_fusion_v2.h"

#include <string>
#include <vector>

#include "utils/ms_context.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "ir/primitive.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/inference_weight_preprocess_utils.h"
#include "utils/shape_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"

namespace mindspore {
namespace opt {
static const auto constexpr kOriQuantOutIdx = 0;
static const auto constexpr kNewQuantOutIdx = 0;
static const auto constexpr kNewScaleOutIdx = 3;

std::vector<std::string> AddRmsNormDynamicQuantV2Fusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimRmsNorm->name(), prim::kPrimAdd->name(), prim::kPrimDynamicQuantExt->name()};
  return ret;
}

const BaseRef AddRmsNormDynamicQuantV2Fusion::DefinePattern() const {
  auto add_rms_norm = VectorRef({prim::kPrimRmsNorm, VectorRef({prim::kPrimAdd, x1_, x2_}), gamma_, eps_});
  auto index0 = std::make_shared<CondVar>(IsConstant);
  auto tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, add_rms_norm, index0});
  auto add_rms_norm_dynamic_quant = VectorRef({prim::kPrimDynamicQuantExt, tuple_get_item_0, smooth_scale_});
  return add_rms_norm_dynamic_quant;
}

// bool IsSupport(const AnfNodePtr &node, const FuncGraphPtr &graph) {
//   auto reshape_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
//   auto rms_norm_out_getitem_0 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(reshape_node), 0);
//   auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_out_getitem_0), 0);
//   auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_node), 0);
//   auto shape1 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 0);
//   auto shape2 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 1);
//   auto rms_x_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(rms_norm_node, 0);
//   auto rms_gamma_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(rms_norm_node, 1);
//   auto smooth_scale_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, 1);
//   FuncGraphManagerPtr mng = graph->manager();
//   MS_EXCEPTION_IF_NULL(mng);

//   auto reshape_users_num = GetRealNodeUsedList(graph, reshape_node)->size();
//   if (reshape_users_num > 1) {
//     MS_LOG(INFO) << "The number of users of reshape_node is more than one: " << reshape_users_num;
//     return false;
//   }

//   const size_t kUserNum = 2;
//   if (mng->node_users()[rms_norm_out_getitem_0].size() != kUserNum) {
//     MS_LOG(INFO) << "The number of users of the rms_norm's first output must be 2, but got: "
//                  << mng->node_users()[rms_norm_out_getitem_0].size();
//     return false;
//   }

//   if (shape1 != shape2) {
//     MS_LOG(INFO) << "AddRmsNormDynamicQuant does not support broadcast, but got shape1: " << shape1
//                  << ", shape2: " << shape2;
//     return false;
//   }

//   if ((rms_x_dtype != kNumberTypeFloat16 && rms_x_dtype != kNumberTypeBFloat16) ||
//       (rms_gamma_dtype != kNumberTypeFloat16 && rms_gamma_dtype != kNumberTypeBFloat16) ||
//       (smooth_scale_dtype != kMetaTypeNone && smooth_scale_dtype != kNumberTypeFloat16 &&
//        smooth_scale_dtype != kNumberTypeBFloat16)) {
//     MS_LOG(INFO) << "Dtype not match, x_dtype: " << rms_x_dtype << ", rms_gamma_dtype: " << rms_gamma_dtype
//                  << ", smooth_scale_dtype: " << smooth_scale_dtype;
//     return false;
//   }

//   auto dynamic_quant_outlist = GetRealNodeUsedList(graph, node);
//   for (const auto &out : *dynamic_quant_outlist) {
//     auto item_index_input = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(out.first), 1);
//     if (!item_index_input->isa<ValueNode>()) {
//       MS_LOG(INFO) << "This is not a valuenode: " << item_index_input->fullname_with_scope();
//       return false;
//     }
//   }

//   // check the user of rstd
//   return true;
// }

static void ReplaceAddResult(const FuncGraphPtr &graph, const FuncGraphManagerPtr &mng,
                             const AnfNodePtr &add_rms_norm_dynamic_quant, const AnfNodePtr &ori_add_node,
                             const std::vector<BaseShapePtr> &add_result_shapes,
                             const std::vector<TypeId> &add_result_types) {
  auto constexpr kNewAddOutIdx = 2;
  auto getitem_for_add = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> add_result_inputs = {NewValueNode(getitem_for_add), add_rms_norm_dynamic_quant,
                                               NewValueNode(static_cast<int64_t>(kNewAddOutIdx))};
  auto add_result = graph->NewCNode(add_result_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, add_result_shapes, add_result.get());
  add_result->set_scope(ori_add_node->scope());
  auto build_info = GenerateKernelBuildInfo(add_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_result.get());
  (void)mng->Replace(ori_add_node, add_result);
}

static void ReplaceDynamicQuantOut(const FuncGraphPtr &graph, const FuncGraphManagerPtr &mng, const AnfNodePtr &node,
                                   const AnfNodePtr &new_quant_getitem, const AnfNodePtr &new_scale_getitem) {
  auto dynamic_quant_outlist = GetRealNodeUsedList(graph, node);
  for (const auto &out : *dynamic_quant_outlist) {
    auto &ori_getitem = out.first;
    auto item_index_input = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(ori_getitem), 1);
    auto item_index_input_value_ptr = item_index_input->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(item_index_input_value_ptr);
    auto idx = GetValue<int64_t>(item_index_input_value_ptr->value());
    if (idx == kOriQuantOutIdx) {
      (void)mng->Replace(ori_getitem, new_quant_getitem);
    } else {
      (void)mng->Replace(ori_getitem, new_scale_getitem);
    }
  }
}

const AnfNodePtr AddRmsNormDynamicQuantV2Fusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                         const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto soc = ms_context->ascend_soc_version();
  if (soc.find("ascend910_93") == std::string::npos && soc.find("ascend910b") == std::string::npos) {
    MS_LOG(INFO) << "AddRmsNormDynamicQuant does not support " << soc;
    return nullptr;
  }

  auto rms_norm_out_getitem_0 = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_out_getitem_0), 0);
  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_node), 0);
  FuncGraphManagerPtr mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);

  // if (!IsSupport(node, graph)) {
  //   return nullptr;
  // }

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[x2_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto smooth_scale1 = utils::cast<AnfNodePtr>((*equiv)[smooth_scale_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);
  auto kernel_graph = graph->cast<KernelGraphPtr>();

  auto quant_out_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  auto quant_out_shape = AnfAlgo::GetOutputDetailShape(node, 0);
  auto scale_out_type = common::AnfAlgo::GetOutputInferDataType(node, 1);
  auto scale_out_shape = AnfAlgo::GetOutputDetailShape(node, 1);
  auto tensor_add_type = common::AnfAlgo::GetOutputInferDataType(tensor_add, 0);
  auto tensor_add_shape = AnfAlgo::GetOutputDetailShape(tensor_add, 0);

  auto prim = std::make_shared<Primitive>("AddRmsNormDynamicQuant");
  auto smooth_scale2 = kernel_graph->NewValueNode(kNone->ToAbstract(), kNone);
  kernel_graph->AddValueNodeToGraph(smooth_scale2);

  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, x2, gamma, smooth_scale1, smooth_scale2, eps};
  auto add_rms_norm_dynamic_quant = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add_rms_norm_dynamic_quant);
  std::vector<TypeId> add_rms_norm_dymaimc_quant_out_types;
  std::vector<BaseShapePtr> add_rms_norm_dymaimc_quant_out_shapes;
  std::vector<TypeId> add_result_types;
  std::vector<BaseShapePtr> add_result_shapes;

  add_rms_norm_dymaimc_quant_out_types.push_back(quant_out_type);
  add_rms_norm_dymaimc_quant_out_shapes.push_back(quant_out_shape);
  add_rms_norm_dymaimc_quant_out_types.push_back(quant_out_type);
  add_rms_norm_dymaimc_quant_out_shapes.push_back(quant_out_shape);
  add_rms_norm_dymaimc_quant_out_types.push_back(tensor_add_type);
  add_rms_norm_dymaimc_quant_out_shapes.push_back(tensor_add_shape);
  add_rms_norm_dymaimc_quant_out_types.push_back(scale_out_type);
  add_rms_norm_dymaimc_quant_out_shapes.push_back(scale_out_shape);
  add_rms_norm_dymaimc_quant_out_types.push_back(scale_out_type);
  add_rms_norm_dymaimc_quant_out_shapes.push_back(scale_out_shape);

  add_result_types.push_back(tensor_add_type);
  add_result_shapes.push_back(tensor_add_shape);

  common::AnfAlgo::SetOutputTypeAndDetailShape(add_rms_norm_dymaimc_quant_out_types,
                                               add_rms_norm_dymaimc_quant_out_shapes, add_rms_norm_dynamic_quant.get());
  add_rms_norm_dynamic_quant->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(add_rms_norm_dynamic_quant);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_rms_norm_dynamic_quant.get());

  ReplaceAddResult(graph, mng, add_rms_norm_dynamic_quant, tensor_add, add_result_shapes, add_result_types);

  auto getitem_for_quant = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> quant_getitem_inputs = {NewValueNode(getitem_for_quant), add_rms_norm_dynamic_quant,
                                                  NewValueNode(static_cast<int64_t>(kNewQuantOutIdx))};
  auto new_quant_item = graph->NewCNode(quant_getitem_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape({quant_out_type}, {quant_out_shape}, new_quant_item.get());
  new_quant_item->set_scope(node->scope());
  auto quant_build_info = GenerateKernelBuildInfo(new_quant_item);
  AnfAlgo::SetSelectKernelBuildInfo(quant_build_info, new_quant_item.get());

  auto getitem_for_scale = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> scale_getitem_inputs = {NewValueNode(getitem_for_scale), add_rms_norm_dynamic_quant,
                                                  NewValueNode(static_cast<int64_t>(kNewScaleOutIdx))};
  auto new_scale_item = graph->NewCNode(scale_getitem_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape({scale_out_type}, {scale_out_shape}, new_scale_item.get());
  new_scale_item->set_scope(node->scope());
  auto scale_build_info = GenerateKernelBuildInfo(new_scale_item);
  AnfAlgo::SetSelectKernelBuildInfo(scale_build_info, new_scale_item.get());

  ReplaceDynamicQuantOut(graph, mng, node, new_quant_item, new_scale_item);

  return add_rms_norm_dynamic_quant;
}
}  // namespace opt
}  // namespace mindspore
