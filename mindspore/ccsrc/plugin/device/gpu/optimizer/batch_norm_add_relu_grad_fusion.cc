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
#include "plugin/device/gpu/optimizer/batch_norm_add_relu_grad_fusion.h"

#include <algorithm>
#include <memory>
#include <vector>
#include <string>

#include "mindspore/ops/op_def/sequence_ops.h"
#include "mindspore/ops/op_def/nn_optimizer_ops.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "ops_utils/op_utils.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "plugin/device/gpu/hal/device/kernel_info_setter.h"
#include "kernel/graph_kernel_info.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_b.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_t.h"

namespace mindspore {
namespace opt {
namespace {
const std::vector<int> kOutputIndex{0, 1, 2};
constexpr size_t kBNGradOutputNum = 3;
constexpr size_t kBNAddReluGradOutputNum = 4;

bool GetBatchNormOutputs(const FuncGraphPtr &func_graph, const AnfNodePtr &bn, std::vector<AnfNodePtr> *bn_outputs) {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(bn);
  MS_EXCEPTION_IF_NULL(bn_outputs);
  auto manager = func_graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  if (manager->node_users().find(bn) == manager->node_users().end()) {
    return false;
  }
  size_t output_num = 0;
  for (const auto &node_index : manager->node_users()[bn]) {
    const AnfNodePtr &output = node_index.first;
    MS_EXCEPTION_IF_NULL(output);
    if (!IsPrimitiveCNode(output, prim::kPrimTupleGetItem)) {
      continue;
    }
    auto tuple_getiterm_cnode = output->cast<CNodePtr>();
    MS_EXCEPTION_IF_NULL(tuple_getiterm_cnode);
    auto index_node = tuple_getiterm_cnode->input(kInputNodeOutputIndexInTupleGetItem);
    MS_EXCEPTION_IF_NULL(index_node);
    auto value_node = index_node->cast<ValueNodePtr>();
    MS_EXCEPTION_IF_NULL(value_node);
    int index = static_cast<int>(GetValue<int64_t>(value_node->value()));
    if (std::find(kOutputIndex.begin(), kOutputIndex.end(), index) == kOutputIndex.end()) {
      return false;
    }
    bn_outputs->push_back(output);
    output_num++;
  }
  return output_num == kBNGradOutputNum;
}

void SetShapeAndType(const CNodePtr &bn_add_relu_grad, const AnfNodePtr &bn_grad, const AnfNodePtr &relu_grad) {
  // set output shape and dtype
  std::vector<TypeId> outputs_type;
  std::vector<BaseShapePtr> outputs_shape;
  auto output_num = AnfAlgo::GetOutputTensorNum(bn_grad);
  for (size_t i = 0; i < output_num; ++i) {
    outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(bn_grad, i));
    outputs_shape.push_back(AnfAlgo::GetOutputDetailShape(bn_grad, i));
  }

  outputs_type.push_back(common::AnfAlgo::GetOutputInferDataType(relu_grad, 0));
  outputs_shape.push_back(AnfAlgo::GetOutputDetailShape(relu_grad, 0));
  common::AnfAlgo::SetOutputTypeAndDetailShape(outputs_type, outputs_shape, bn_add_relu_grad.get());
}

void ReplaceOutput(const FuncGraphPtr &graph, const AnfNodePtr &bn_grad, const AnfNodePtr &relu_grad,
                   const CNodePtr &bn_add_relu_grad) {
  // Create outputs
  std::vector<AnfNodePtr> bn_add_relu_grad_output;
  CreateMultipleOutputsOfAnfNode(graph, bn_add_relu_grad, kBNAddReluGradOutputNum, &bn_add_relu_grad_output);
  if (bn_add_relu_grad_output.size() != kBNAddReluGradOutputNum) {
    MS_LOG(EXCEPTION) << "The output size of node " << kBatchNormGradWithAddAndActivationOpName << " must be "
                      << kBNAddReluGradOutputNum << ", but it is " << bn_add_relu_grad_output.size();
  }

  // Get bn outputs
  std::vector<AnfNodePtr> bn_outputs;
  if (!GetBatchNormOutputs(graph, bn_grad, &bn_outputs)) {
    MS_LOG(INFO) << "The " << prim::kPrimBatchNormGrad
                 << " node should only have output 0, 1 and 2. The node should not be changed";
    return;
  }

  // Replace original output
  auto manager = graph->manager();
  MS_EXCEPTION_IF_NULL(manager);
  sort(bn_outputs.begin(), bn_outputs.end(), CompareTupleGetitem);
  size_t output_index = 0;
  for (const auto &output : bn_outputs) {
    (void)manager->Replace(output, bn_add_relu_grad_output[output_index]);
    output_index++;
  }

  if (!manager->Replace(relu_grad, bn_add_relu_grad_output[kBNAddReluGradOutputNum - 1])) {
    MS_LOG(EXCEPTION) << "manager replace node failed in batchnorm add relu grad fusion.";
  }
  return;
}

bool PatternCheck(const FuncGraphPtr &graph, const AnfNodePtr &node) {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  size_t format_idx = ops::GetInputIndexByName(common::AnfAlgo::GetCNodeName(node), "data_format");
  if (format_idx == SIZE_MAX) {
    return false;
  }
  auto format_input_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), format_idx);
  if (!utils::isa<ValueNodePtr>(format_input_node)) {
    return false;
  }
  auto format_v = GetScalarValue<int64_t>(format_input_node->cast<ValueNodePtr>()->value());
  if (!format_v.has_value()) {
    return false;
  }
  if (AnfAlgo::GetInputFormat(node, 0) != kOpFormat_NHWC && format_v.value() != Format::NHWC) {
    return false;
  }

  auto shape = AnfAlgo::GetInputDeviceShape(node, 0);
  if ((shape.back() % kBNChannelMultipleFactor) != 0) {
    return false;
  }

  auto relu_grad = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), 0);
  MS_EXCEPTION_IF_NULL(relu_grad);
  auto relu_users = GetRealNodeUsedList(graph, relu_grad);
  if (relu_users->size() != 2) {
    return false;
  }

  // process pattern as Relu(TensorAdd(BN#0, BN#1))
  auto tuple_getitem = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex5);
  MS_EXCEPTION_IF_NULL(tuple_getitem);
  if (!utils::isa<CNodePtr>(tuple_getitem) ||
      common::AnfAlgo::GetCNodeName(tuple_getitem) != prim::kPrimTupleGetItem->name()) {
    return false;
  }
  auto forward_node = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(tuple_getitem), 0);
  if (common::AnfAlgo::GetCNodeName(forward_node) != kBatchNormWithAddAndActivationOpName) {
    return false;
  }

  return true;
}
}  // namespace

const BaseRef BatchNormAddReluGradFusion::DefinePattern() const {
  VectorRef relu_grad = VectorRef({prim::kPrimReluGrad, dy_, y_});
  VectorRef batch_norm_grad = VectorRef(
    {prim::kPrimBatchNormGrad, relu_grad, x_, scale_, save_mean_, save_var_, reserve_, is_training_, eps_, format_});
  return batch_norm_grad;
}

const AnfNodePtr BatchNormAddReluGradFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                     const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);

  if (!PatternCheck(graph, node)) {
    return nullptr;
  }

  auto relu_grad = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex0);
  MS_EXCEPTION_IF_NULL(relu_grad);
  auto dy = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(relu_grad), kIndex0);
  MS_EXCEPTION_IF_NULL(dy);
  auto y = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(relu_grad), kIndex1);
  MS_EXCEPTION_IF_NULL(y);
  auto x = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex1);
  MS_EXCEPTION_IF_NULL(x);
  auto scale = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex2);
  MS_EXCEPTION_IF_NULL(scale);
  auto save_mean = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex3);
  MS_EXCEPTION_IF_NULL(save_mean);
  auto save_var = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex4);
  MS_EXCEPTION_IF_NULL(save_var);
  auto reserve = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex5);
  MS_EXCEPTION_IF_NULL(reserve);
  auto is_train = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex6);
  MS_EXCEPTION_IF_NULL(is_train);
  auto eps = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex7);
  MS_EXCEPTION_IF_NULL(eps);
  auto format = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(node), kIndex8);
  MS_EXCEPTION_IF_NULL(format);
  auto batch_norm = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(save_mean), kIndex0);
  MS_EXCEPTION_IF_NULL(batch_norm);
  auto bias = common::AnfAlgo::GetInputNode(utils::cast<CNodePtr>(batch_norm), kIndex2);
  MS_EXCEPTION_IF_NULL(bias);
  if (!utils::isa<ValueNodePtr>(is_train)) {
    return nullptr;
  }
  auto is_train_v = GetScalarValue<bool>(is_train->cast<ValueNodePtr>()->value());
  if (!is_train_v.has_value() || !is_train_v.value()) {
    return nullptr;
  }

  auto prim = std::make_shared<Primitive>(kBatchNormGradWithAddAndActivationOpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), dy,  x,     scale, save_mean, save_var, reserve, bias, y,
                                    is_train,           eps, format};
  auto fused_batch_norm_add_relu_grad = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(fused_batch_norm_add_relu_grad);
  common::AnfAlgo::CopyNodeAttrs(node, fused_batch_norm_add_relu_grad);
  SetShapeAndType(fused_batch_norm_add_relu_grad, node, relu_grad);
  ReplaceOutput(graph, node, relu_grad, fused_batch_norm_add_relu_grad);
  auto kernel_info_setter = GraphKernelInfoManager::Instance().GetGraphKernelInfo(kGPUDevice);
  kernel_info_setter->SetKernelInfo(fused_batch_norm_add_relu_grad, KernelType::UNKNOWN_KERNEL_TYPE);
  return nullptr;
}
}  // namespace opt
}  // namespace mindspore
