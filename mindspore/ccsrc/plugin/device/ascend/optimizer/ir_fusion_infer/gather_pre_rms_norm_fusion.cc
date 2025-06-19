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
#include "plugin/device/ascend/optimizer/ir_fusion_infer/gather_pre_rms_norm_fusion.h"

#include <string>
#include <vector>
#include <set>

#include "backend/common/pass/common/gllo_utils.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "include/backend/optimizer/helper.h"
#include "ir/primitive.h"
#include "utils/shape_utils.h"
#include "utils/ms_context.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_d.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_g.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_q.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
std::vector<std::string> GatherPreRmsNormFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimRmsNorm->name(), prim::kPrimAdd->name(), prim::kPrimGather->name()};
  return ret;
}

const BaseRef GatherPreRmsNormFusion::DefinePattern() const {
  VarPtr axis = std::make_shared<CondVar>(IsConstant);
  VarPtr batch_dims = std::make_shared<CondVar>(IsConstant);
  auto gather = VectorRef({prim::kPrimGather, res_in_, indices_, axis, batch_dims});
  auto add = VectorRef({prim::kPrimAdd, x_, gather});
  auto rms_norm = VectorRef({prim::kPrimRmsNorm, add, gamma_, epsilon_});
  auto index0 = std::make_shared<CondVar>(IsConstant);
  auto tuple_get_item = VectorRef({prim::kPrimTupleGetItem, rms_norm, index0});
  return tuple_get_item;
}

CNodePtr GatherPreRmsNormFusion::CreateGatherPreRmsNormNode(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                            const EquivPtr &equiv) const {
  MS_LOG(DEBUG) << "start create GatherPreRmsNorm node";
  MS_ASSERT(graph != nullptr && node != nullptr && equiv != nullptr);

  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_node), 0);
  FuncGraphManagerPtr mng = graph->manager();
  MS_EXCEPTION_IF_NULL(mng);

  auto x = utils::cast<AnfNodePtr>((*equiv)[x_]);
  auto res_in = utils::cast<AnfNodePtr>((*equiv)[res_in_]);
  auto indices = utils::cast<AnfNodePtr>((*equiv)[indices_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto epsilon = utils::cast<AnfNodePtr>((*equiv)[epsilon_]);

  const std::set<TypeId> support_dtype = {kNumberTypeFloat16, kNumberTypeBFloat16};
  if (!CheckSupportDataType(x, support_dtype)) {
    return nullptr;
  }

  auto y_type = common::AnfAlgo::GetOutputInferDataType(node, 0);
  auto y_shape = AnfAlgo::GetOutputDetailShape(node, 0);
  auto res_out_type = common::AnfAlgo::GetOutputInferDataType(tensor_add, 0);
  auto res_out_shape = AnfAlgo::GetOutputDetailShape(tensor_add, 0);

  auto prim = std::make_shared<Primitive>("GatherPreRmsNorm");
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x, res_in, indices, gamma, epsilon};
  auto gather_pre_rms_norm = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(gather_pre_rms_norm);

  std::vector<TypeId> gather_pre_rms_norm_out_types;
  std::vector<BaseShapePtr> gather_pre_rms_norm_out_shapes;
  std::vector<TypeId> rms_result_types;
  std::vector<BaseShapePtr> rms_result_shapes;
  std::vector<TypeId> add_result_types;
  std::vector<BaseShapePtr> add_result_shapes;

  gather_pre_rms_norm_out_types.push_back(kNumberTypeFloat32);
  gather_pre_rms_norm_out_shapes.push_back(y_shape);
  gather_pre_rms_norm_out_types.push_back(res_out_type);
  gather_pre_rms_norm_out_shapes.push_back(res_out_shape);

  rms_result_types.push_back(kNumberTypeFloat32);
  rms_result_shapes.push_back(y_shape);

  add_result_types.push_back(res_out_type);
  add_result_shapes.push_back(res_out_shape);

  common::AnfAlgo::SetOutputTypeAndDetailShape(gather_pre_rms_norm_out_types, gather_pre_rms_norm_out_shapes,
                                               gather_pre_rms_norm.get());
  gather_pre_rms_norm->set_scope(node->scope());
  auto build_info = GenerateKernelBuildInfo(gather_pre_rms_norm);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, gather_pre_rms_norm.get());

  // y
  auto constexpr kNewRmsOutIdx = 0;
  auto getitem_for_rms = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> rms_result_inputs = {NewValueNode(getitem_for_rms), gather_pre_rms_norm,
                                               NewValueNode(static_cast<int64_t>(kNewRmsOutIdx))};
  auto rms_result = graph->NewCNode(rms_result_inputs);
  MS_EXCEPTION_IF_NULL(rms_result);
  common::AnfAlgo::SetOutputTypeAndDetailShape(rms_result_types, rms_result_shapes, rms_result.get());
  rms_result->set_scope(rms_norm_node->scope());
  build_info = GenerateKernelBuildInfo(rms_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, rms_result.get());
  auto cast_result = AddCastNode(graph, y_type, rms_result, false);

  // res_out
  auto constexpr kNewAddOutIdx = 1;
  auto getitem_for_add = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> add_result_inputs = {NewValueNode(getitem_for_add), gather_pre_rms_norm,
                                               NewValueNode(static_cast<int64_t>(kNewAddOutIdx))};
  auto add_result = graph->NewCNode(add_result_inputs);
  MS_EXCEPTION_IF_NULL(add_result);
  common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, add_result_shapes, add_result.get());
  add_result->set_scope(tensor_add->scope());
  build_info = GenerateKernelBuildInfo(add_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_result.get());
  (void)mng->Replace(tensor_add, add_result);

  return cast_result;
}

const AnfNodePtr GatherPreRmsNormFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                 const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (!ms_context->IsEnableInferBoost()) {
    return nullptr;
  }

  auto soc = ms_context->ascend_soc_version();
  if (!soc.empty() && soc.find("ascend910_93") == std::string::npos && soc.find("ascend910b") == std::string::npos) {
    MS_LOG(INFO) << "GatherPreRmsNorm does not support " << soc;
    return nullptr;
  }

  auto cnode = CreateGatherPreRmsNormNode(graph, node, equiv);
  if (cnode == nullptr) {
    MS_LOG(DEBUG) << "create gather_pre_rms_norm node failed.";
    return nullptr;
  }
  return cnode;
}
}  // namespace opt
}  // namespace mindspore
