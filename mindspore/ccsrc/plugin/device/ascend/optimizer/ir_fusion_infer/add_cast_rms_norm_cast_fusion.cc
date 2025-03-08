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
#include "plugin/device/ascend/optimizer/ir_fusion_infer/add_cast_rms_norm_cast_fusion.h"

#include <vector>
#include <string>

#include "utils/ms_context.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/backend/optimizer/helper.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
std::vector<std::string> AddCastRmsNormCastFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimRmsNorm->name(), prim::kPrimCast->name(), prim::kPrimAdd->name()};
  return ret;
}

const BaseRef AddCastRmsNormCastFusion::DefinePattern() const {
  VarPtr index0 = std::make_shared<CondVar>(IsConstant);
  VarPtr x0 = std::make_shared<Var>();
  VarPtr x1 = std::make_shared<Var>();
  VectorRef add_cast = VectorRef({prim::kPrimCast, VectorRef({prim::kPrimAdd, x1_, x2_}), x0});
  VectorRef add_cast_rms_norm = VectorRef({prim::kPrimRmsNorm, add_cast, gamma_, eps_});
  VectorRef tuple_get_item_0 = VectorRef({prim::kPrimTupleGetItem, add_cast_rms_norm, index0});
  VectorRef add_cast_rms_norm_cast = VectorRef({prim::kPrimCast, tuple_get_item_0, x1});
  return add_cast_rms_norm_cast;
}

const AnfNodePtr AddCastRmsNormCastFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const std::string fusion_op_name = "AddRmsNorm";
  auto &enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  bool enable_add_rmsnorm =
    (std::find(enable_op_list.begin(), enable_op_list.end(), fusion_op_name) != enable_op_list.end());
  if (!enable_add_rmsnorm) {
    return nullptr;
  }

  auto x1 = utils::cast<AnfNodePtr>((*equiv)[x1_]);
  auto x2 = utils::cast<AnfNodePtr>((*equiv)[x2_]);
  auto gamma = utils::cast<AnfNodePtr>((*equiv)[gamma_]);
  auto eps = utils::cast<AnfNodePtr>((*equiv)[eps_]);

  auto tuple_get_item_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  auto rms_norm_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tuple_get_item_node), 0);
  if (IsUsedByOthers(graph, rms_norm_node)) {
    // we only deal with case that rstd result from rms norm is not used
    return nullptr;
  }

  auto cast_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(rms_norm_node), 0);
  auto tensor_add = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(cast_node), 0);
  MS_EXCEPTION_IF_NULL(tensor_add);

  auto shape1 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 0);
  auto shape2 = common::AnfAlgo::GetPrevNodeOutputInferShape(tensor_add, 1);
  if (shape1 != shape2) {
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>("AddRmsNorm");
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), x1, x2, gamma, eps};
  auto add_rms_norm = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(add_rms_norm);

  std::vector<TypeId> types;
  std::vector<BaseShapePtr> shapes;

  // the first output is the result after cast
  std::vector<TypeId> cast_result_types;
  std::vector<BaseShapePtr> cast_result_shapes;
  cast_result_types.push_back(common::AnfAlgo::GetOutputInferDataType(node, 0));
  cast_result_shapes.push_back(AnfAlgo::GetOutputDetailShape(node, 0));
  types.push_back(common::AnfAlgo::GetOutputInferDataType(node, 0));
  shapes.push_back(AnfAlgo::GetOutputDetailShape(node, 0));

  // the second output is rstd from the second output of rmsnorm
  size_t output_num = AnfAlgo::GetOutputElementNum(rms_norm_node);
  constexpr auto kRmsNormOutputNum = 2;
  MS_EXCEPTION_IF_CHECK_FAIL(output_num == kRmsNormOutputNum,
                             "For rms norm, it should have 2 output, but get: " + output_num);

  types.push_back(common::AnfAlgo::GetOutputInferDataType(rms_norm_node, 1));
  shapes.push_back(AnfAlgo::GetOutputDetailShape(rms_norm_node, 1));

  // the third output is the add output

  std::vector<TypeId> add_result_types;
  std::vector<BaseShapePtr> add_result_shapes;
  add_result_types.push_back(common::AnfAlgo::GetOutputInferDataType(tensor_add, 0));
  add_result_shapes.push_back(AnfAlgo::GetOutputDetailShape(tensor_add, 0));

  types.push_back(add_result_types[0]);
  shapes.push_back(add_result_shapes[0]);

  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, add_rms_norm.get());
  add_rms_norm->set_scope(rms_norm_node->scope());

  auto build_info = GenerateKernelBuildInfo(add_rms_norm);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_rms_norm.get());

  // the first element of add_rms_norm result is the original result
  auto prim_getitem = std::make_shared<Primitive>("TupleGetItem");
  std::vector<AnfNodePtr> cast_result_inputs = {NewValueNode(prim_getitem), add_rms_norm,
                                                NewValueNode(static_cast<int64_t>(0))};
  auto cast_result = graph->NewCNode(cast_result_inputs);

  common::AnfAlgo::SetOutputTypeAndDetailShape(cast_result_types, cast_result_shapes, cast_result.get());
  cast_result->set_scope(node->scope());

  std::vector<AnfNodePtr> add_result_inputs = {NewValueNode(prim_getitem), add_rms_norm,
                                               NewValueNode(static_cast<int64_t>(2))};
  auto add_result = graph->NewCNode(add_result_inputs);

  common::AnfAlgo::SetOutputTypeAndDetailShape(add_result_types, add_result_shapes, add_result.get());
  add_result->set_scope(tensor_add->scope());

  build_info = GenerateKernelBuildInfo(add_result);
  AnfAlgo::SetSelectKernelBuildInfo(build_info, add_result.get());

  FuncGraphManagerPtr manager = graph->manager();
  (void)manager->Replace(tensor_add, add_result);

  return cast_result;
}
}  // namespace opt
}  // namespace mindspore
