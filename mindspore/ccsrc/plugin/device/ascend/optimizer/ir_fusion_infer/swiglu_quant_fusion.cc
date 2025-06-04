
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
#include "plugin/device/ascend/optimizer/ir_fusion_infer/swiglu_quant_fusion.h"

#include <string>
#include <vector>
#include <set>

#include "backend/common/pass/common/gllo_utils.h"
#include "utils/ms_context.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "ir/primitive.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"
#include "utils/shape_utils.h"
#include "plugin/device/ascend/optimizer/ir_fusion_infer/inference_weight_preprocess_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_q.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int64_t kMaxDim = 8192;
AnfNodePtr NewGroupIndexInput(const FuncGraphPtr &func_graph, const AnfNodePtr &input_node) {
  auto x_abs = input_node->abstract();
  auto x_shape_ptr = x_abs->GetShape();
  auto x_shape = x_shape_ptr->GetShapeVector();
  std::vector<int32_t> group_index_val{static_cast<int32_t>(x_shape[kIndex0])};
  auto group_index_tensor = std::make_shared<tensor::Tensor>(group_index_val);
  auto group_index_node = CreateValueNodeWithKernelInfo(func_graph, group_index_tensor);
  return group_index_node;
}

CNodePtr NewGetItemNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const CNodePtr &input_cnode,
                        const std::vector<TypeId> &out_dtypes, std::vector<BaseShapePtr> *out_shapes) {
  auto prim_getitem = std::make_shared<Primitive>(kTupleGetItemOpName);
  std::vector<AnfNodePtr> getitem_inputs = {NewValueNode(prim_getitem), input_cnode,
                                            NewValueNode(static_cast<int64_t>(kIndex0))};
  auto output_cnode = func_graph->NewCNode(getitem_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape(out_dtypes, *out_shapes, output_cnode.get());
  output_cnode->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(output_cnode);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, output_cnode.get());
  return output_cnode;
}

AnfNodePtr NewReshapedNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node, const AnfNodePtr &input_node) {
  auto input_shape_vec = common::AnfAlgo::GetOutputInferShape(input_node, kIndex0);
  if (input_shape_vec.size() == kSizeTwo && input_shape_vec[kIndex0] == 1) {
    return input_node;
  }
  auto reshape_prim = std::make_shared<Primitive>(kReshapeOpName);
  ShapeVector new_shape{1, abstract::Shape::kShapeDimAny};
  auto out_shape = std::make_shared<abstract::Shape>(new_shape);
  std::vector<AnfNodePtr> reshape_inputs = {NewValueNode(reshape_prim), input_node,
                                            CreateShapeValueNode(func_graph, new_shape, false)};
  auto reshaped_out = func_graph->NewCNode(reshape_inputs);
  common::AnfAlgo::SetOutputTypeAndDetailShape({kNumberTypeFloat32}, {out_shape}, reshaped_out.get());
  reshaped_out->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(reshaped_out);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, reshaped_out.get());

  return reshaped_out;
}

bool IsSupport(const FuncGraphPtr &func_graph, const AnfNodePtr &node) {
  auto swiglu_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex0);
  auto x_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(swiglu_node), kIndex0);
  auto axis_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(swiglu_node), kIndex1);

  auto offset_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, kIndex2);
  if (offset_dtype != kNumberTypeFloat32) {
    MS_LOG(DEBUG) << "dtype of offset is not float32";
    return false;
  }

  if (!axis_node->isa<ValueNode>()) {
    MS_LOG(DEBUG) << "axis node is not a value node";
    return false;
  }

  auto x_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(swiglu_node, kIndex0);
  if (x_shape.back() > kMaxDim) {
    MS_LOG(DEBUG) << "the last dim of x must <= " << kMaxDim;
    return false;
  }

  const std::set<TypeId> support_x_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32};
  if (!CheckSupportDataType(x_node, support_x_dtype)) {
    MS_LOG(DEBUG) << "input x dtype is not support";
    return false;
  }
  return true;
}

}  // namespace

std::vector<std::string> SwigluQuantFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimQuantV2->name(), prim::kPrimSwiglu->name()};
  return ret;
}

const BaseRef SwigluQuantFusion::DefinePattern() const {
  auto is_swiglu = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimSwiglu>);
  MS_CHECK_TRUE_RET(is_swiglu != nullptr, {});
  VectorRef swiglu_ref({is_swiglu, x_, axis_});
  auto is_static_quant = std::make_shared<CondVar>(IsSpecifiedNode<&prim::kPrimQuantV2>);
  MS_CHECK_TRUE_RET(is_static_quant != nullptr, {});
  auto sqrt_mode = std::make_shared<CondVar>(IsConstant);      // not used
  auto rounding_mode = std::make_shared<CondVar>(IsConstant);  // not used
  auto dst_type = std::make_shared<CondVar>(IsConstant);       // not used
  VectorRef static_quant_ref({is_static_quant, swiglu_ref, smooth_scale_, offset_, sqrt_mode, rounding_mode, dst_type});
  return static_quant_ref;
}

CNodePtr SwigluQuantFusion::CreateSwigluQuantNode(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "start create SwigluQuant node";
  MS_ASSERT(func_graph != nullptr && node != nullptr && equiv != nullptr);
  std::string prim_name = "SwigluQuant";
  auto swiglu_quant_prim = std::make_shared<Primitive>(prim_name);
  auto x_node = utils::cast<AnfNodePtr>((*equiv)[x_]);
  MS_EXCEPTION_IF_NULL(x_node);
  auto smooth_scale_node = utils::cast<AnfNodePtr>((*equiv)[smooth_scale_]);
  MS_EXCEPTION_IF_NULL(smooth_scale_node);
  auto offset_node = utils::cast<AnfNodePtr>((*equiv)[offset_]);
  MS_EXCEPTION_IF_NULL(offset_node);

  auto smooth_scale_dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(node, kIndex1);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  if (smooth_scale_dtype != kNumberTypeFloat32) {
    smooth_scale_node = ConvertWeightsToNewType(smooth_scale_node);
    kernel_graph->AddValueNodeToGraph(smooth_scale_node->cast<ValueNodePtr>());
  }

  smooth_scale_node = NewReshapedNode(func_graph, node, smooth_scale_node);
  offset_node = NewReshapedNode(func_graph, node, offset_node);
  auto group_index_node = NewGroupIndexInput(func_graph, x_node);
  auto activate_left_node = CreateValueNodeWithKernelInfo(func_graph, MakeValue<bool>(true));
  auto quant_mode_node = CreateValueNodeWithKernelInfo(func_graph, MakeValue<int64_t>(0));

  std::vector<AnfNodePtr> quant_inputs = {x_node,           smooth_scale_node,  offset_node,
                                          group_index_node, activate_left_node, quant_mode_node};
  auto swiglu_quant = func_graph->NewCNode(swiglu_quant_prim, quant_inputs);
  MS_EXCEPTION_IF_NULL(swiglu_quant);

  std::vector<TypeId> swiglu_quant_out_types;
  std::vector<BaseShapePtr> swiglu_quant_out_shapes;
  std::vector<TypeId> quant_out_types;
  std::vector<BaseShapePtr> quant_out_shapes;
  auto quant_out_type = common::AnfAlgo::GetOutputInferDataType(node, kIndex0);
  auto quant_out_shape = AnfAlgo::GetOutputDetailShape(node, kIndex0);
  quant_out_types.push_back(quant_out_type);
  quant_out_shapes.push_back(quant_out_shape);
  auto quant_out_shape_vec = quant_out_shape->GetShapeVector();
  auto out_scale_shape = quant_out_shape->Clone();
  ShapeVector out_scale_shape_vec(quant_out_shape_vec.begin(), quant_out_shape_vec.end() - kSizeOne);
  out_scale_shape->SetShapeVector(out_scale_shape_vec);
  TypeId out_scale_type = kNumberTypeFloat32;
  swiglu_quant_out_types.push_back(quant_out_type);
  swiglu_quant_out_shapes.push_back(quant_out_shape);
  swiglu_quant_out_types.push_back(out_scale_type);
  swiglu_quant_out_shapes.push_back(out_scale_shape);

  common::AnfAlgo::SetOutputTypeAndDetailShape(swiglu_quant_out_types, swiglu_quant_out_shapes, swiglu_quant.get());
  swiglu_quant->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(swiglu_quant);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, swiglu_quant.get());

  auto swiglu_quant_result = NewGetItemNode(func_graph, node, swiglu_quant, quant_out_types, &quant_out_shapes);

  MS_LOG(DEBUG) << "create SwigluQuant node success.";
  return swiglu_quant_result;
}

const AnfNodePtr SwigluQuantFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                            const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  auto soc = ms_context->ascend_soc_version();
  if (!soc.empty() && soc.find("ascend910b") == std::string::npos) {
    MS_LOG(DEBUG) << "SwigluQuant does not support " << soc;
    return nullptr;
  }

  if (!IsSupport(graph, node)) {
    return nullptr;
  }

  auto cnode = CreateSwigluQuantNode(graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "create swiglu quant node failed.";
    return nullptr;
  }
  MS_LOG(DEBUG) << "Process SwigluQuant fusion success.";
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
