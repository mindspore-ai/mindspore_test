/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "backend/common/pass/ir_fusion/adaptive_max_pool2d_fusion.h"
#include <memory>
#include <vector>
#include <string>
#include "mindspore/ops/op_def/conv_pool_ops.h"
#include "include/backend/optimizer/helper.h"
#include "abstract/dshape.h"
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/kernel_graph.h"
#include "include/common/utils/anfalgo.h"
#include "ir/anf.h"
#include "ir/func_graph.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/type_id.h"
#include "utils/log_adapter.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"

namespace mindspore {
namespace opt {
namespace {
constexpr int64_t kMaxPaddingSize = 10;
constexpr int64_t kAdaptiveMaxpool2DOutputNumber = 1;

std::vector<int64_t> ComputeKernelAttr(int64_t input_size, int64_t output_size) {
  int64_t padding_size = 0;
  int64_t kernel_size = 0;
  int64_t ceil_mode = 1;
  const int64_t double_val = 2;
  const int64_t pad_default_val = 2;

  if (input_size % output_size == 0) {
    kernel_size = input_size / output_size;
    int64_t stride_size = kernel_size;
    padding_size = 0;
    return std::vector<int64_t>{0, kernel_size, stride_size, padding_size, ceil_mode};
  }

  int64_t div_size = input_size / output_size;
  for (kernel_size = div_size + 1; kernel_size <= div_size + pad_default_val; ++kernel_size) {
    for (int64_t stride_size = 1; stride_size < kernel_size; ++stride_size) {
      int64_t res0 = (input_size + double_val * padding_size - kernel_size) / stride_size + 1;
      if (res0 == output_size) {
        return std::vector<int64_t>{1, kernel_size, stride_size, padding_size, ceil_mode};
      }
    }
  }

  while (padding_size >= 0) {
    for (kernel_size = padding_size + 1; kernel_size <= input_size + pad_default_val * padding_size; ++kernel_size) {
      for (int64_t stride_size = 1; stride_size <= kernel_size; ++stride_size) {
        int64_t res0 = (input_size + double_val * padding_size - kernel_size) / stride_size + 1;
        if (res0 == output_size) {
          return std::vector<int64_t>{pad_default_val, kernel_size, stride_size, padding_size, ceil_mode};
        }
      }
    }
    padding_size++;
    if (padding_size >= kMaxPaddingSize) {
      break;
    }
  }
  return std::vector<int64_t>{-1, -1, -1, -1, ceil_mode};
}

void SetNodeAttr(const AnfNodePtr &node, const std::vector<int64_t> &height_attr,
                 const std::vector<int64_t> &width_attr) {
  std::vector<int64_t> window_value{height_attr[kIndex1], width_attr[kIndex1]};
  std::vector<int64_t> stride_value{height_attr[kIndex2], width_attr[kIndex2]};
  std::vector<int64_t> pad_value{height_attr[kIndex3], height_attr[kIndex3], width_attr[kIndex3], width_attr[kIndex3]};
  int64_t ceil_mode_value = 1;
  int64_t mode_value = 0;
  std::vector<int64_t> dilation_value = {1, 1, 1, 1};

  common::AnfAlgo::SetNodeAttr(kAttrWindow, MakeValue(window_value), node);
  common::AnfAlgo::SetNodeAttr(kAttrStride, MakeValue(stride_value), node);
  common::AnfAlgo::SetNodeAttr(kAttrMode, MakeValue(mode_value), node);
  common::AnfAlgo::SetNodeAttr(kAttrPad, MakeValue(pad_value), node);
  common::AnfAlgo::SetNodeAttr(kAttrGlobalPooling, MakeValue(false), node);
  common::AnfAlgo::SetNodeAttr(kAttrCeilMode, MakeValue(ceil_mode_value), node);
  common::AnfAlgo::SetNodeAttr(kAttrDilation, MakeValue(dilation_value), node);
  common::AnfAlgo::SetNodeAttr(kAttrDatFormat, MakeValue("NCHW"), node);
}
}  // namespace

std::vector<std::string> AdaptiveMaxPool2DGeFusion::MustExistPrimitiveName() const {
  std::vector<std::string> ret{prim::kPrimAdaptiveMaxPool2D->name()};
  return ret;
}

const BaseRef AdaptiveMaxPool2DFusion::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  return VectorRef({prim::kPrimAdaptiveMaxPool2d, X});
}

const BaseRef AdaptiveMaxPool2DGeFusion::DefinePattern() const {
  VarPtr X = std::make_shared<Var>();
  VarPtr Y = std::make_shared<Var>();
  return VectorRef({prim::kPrimAdaptiveMaxPool2D, X, Y});
}

const AnfNodePtr AdaptiveMaxPool2DFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                  const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto adaptive_max_pool2d = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(adaptive_max_pool2d);
  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(adaptive_max_pool2d, kIndex0);
  if (input_shape.size() != kShape3dDims && input_shape.size() != kShape4dDims) {
    MS_LOG(EXCEPTION) << "AdaptiveMaxPool2D's input shape must equal to 3 or 4, but got " << input_shape.size();
  }

  // process output_size
  if (!common::AnfAlgo::HasNodeAttr(kAttrOutputSize, adaptive_max_pool2d)) {
    MS_LOG(EXCEPTION) << "AdaptiveMaxPool2D need to set output_size attribute.";
  }
  auto output_size = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(node, kAttrOutputSize);
  if (output_size.size() != kShape2dDims) {
    MS_LOG(EXCEPTION) << "AdaptiveMaxPool2D's output_size shape should equal to 2.";
  }

  size_t w_index = input_shape.size() - kIndex1;
  size_t h_index = input_shape.size() - kIndex2;
  int64_t height = input_shape.at(h_index);
  int64_t width = input_shape.at(w_index);
  int64_t output_h = (output_size[kDim0] == -1) ? height : output_size[kDim0];
  int64_t output_w = (output_size[kDim1] == -1) ? width : output_size[kDim1];
  if ((output_h != -1 && output_h <= 0) || (output_w != -1 && output_w <= 0)) {
    MS_LOG(EXCEPTION) << "AdaptiveMaxPool2D's output_size value is invalid.";
  }
  std::vector<int64_t> new_output_size{output_h, output_w};
  common::AnfAlgo::SetNodeAttr(kAttrOutputSize, MakeValue(new_output_size), adaptive_max_pool2d);

  if (AnfAlgo::GetOutputElementNum(adaptive_max_pool2d) > 1) {
    return nullptr;
  }

  if (height % output_h != 0 || width % output_w != 0) {
    auto types = {common::AnfAlgo::GetOutputInferDataType(adaptive_max_pool2d, 0), kNumberTypeInt64};
    auto shapes = {AnfAlgo::GetOutputDetailShape(adaptive_max_pool2d, 0),
                   AnfAlgo::GetOutputDetailShape(adaptive_max_pool2d, 0)};
    common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, adaptive_max_pool2d.get());
    std::vector<AnfNodePtr> multi_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, adaptive_max_pool2d, kAdaptiveMaxpool2DOutputNumber, &multi_outputs);
    return multi_outputs[kIndex0];
  }
  auto height_attr = ComputeKernelAttr(height, output_h);
  auto width_attr = ComputeKernelAttr(width, output_w);
  if (height_attr[kIndex0] == -1 || width_attr[kIndex0] == -1) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Current AdaptiveMaxPool2D not support this scene! node:"
                                      << node->DebugString();
  }

  std::vector<AnfNodePtr> pooling_inputs = {NewValueNode(std::make_shared<Primitive>(kPoolingOpName))};
  (void)pooling_inputs.insert(pooling_inputs.end(), adaptive_max_pool2d->inputs().begin() + 1,
                              adaptive_max_pool2d->inputs().end());
  auto pooling = NewCNode(pooling_inputs, kernel_graph);
  auto types = {common::AnfAlgo::GetOutputInferDataType(adaptive_max_pool2d, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(adaptive_max_pool2d, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, pooling.get());
  pooling->set_scope(adaptive_max_pool2d->scope());
  SetNodeAttr(pooling, height_attr, width_attr);
  return pooling;
}

const bool IsDynamicShapeGe(const CNodePtr &adaptive_max_pool2d) {
  auto origin_primitive = GetCNodePrimitive(adaptive_max_pool2d);
  MS_EXCEPTION_IF_NULL(origin_primitive);
  if (common::AnfAlgo::IsDynamicShape(adaptive_max_pool2d)) {
    MS_LOG(INFO) << "Exit adaptive_max_pool2d pass, due to dynamic attr:" << origin_primitive->GetAttrsText();
    return true;
  }

  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(adaptive_max_pool2d, kIndex0);
  if (IsDynamicShape(input_shape) || IsDynamicRank(input_shape)) {
    MS_LOG(INFO) << "Exit adaptive_max_pool2d pass due to dynamic input shape.";
    return true;
  }

  return false;
}

const AnfNodePtr AdaptiveMaxPool2DGeFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                    const EquivPtr &) const {
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_EXCEPTION_IF_NULL(node);
  auto adaptive_max_pool2d = node->cast<CNodePtr>();
  MS_EXCEPTION_IF_NULL(adaptive_max_pool2d);
  if (IsDynamicShapeGe(adaptive_max_pool2d)) {
    return node;
  }

  auto kernel_graph = func_graph->cast<KernelGraphPtr>();
  MS_EXCEPTION_IF_NULL(kernel_graph);

  auto input_shape = common::AnfAlgo::GetPrevNodeOutputInferShape(adaptive_max_pool2d, kIndex0);
  if (input_shape.size() != kShape3dDims && input_shape.size() != kShape4dDims) {
    MS_LOG(EXCEPTION) << "AdaptiveMaxPool2D's input shape must equal to 3 or 4, but got " << input_shape.size();
  }

  auto output_size_node = common::AnfAlgo::GetInputNode(adaptive_max_pool2d, kIndex1);
  MS_EXCEPTION_IF_NULL(output_size_node);
  auto output_size_abstract = output_size_node->abstract();
  MS_EXCEPTION_IF_NULL(output_size_abstract);
  auto output_size_val = output_size_abstract->GetValue();
  MS_EXCEPTION_IF_NULL(output_size_val);
  const auto output_size_opt = GetArrayValue<int64_t>(output_size_val);
  if (!output_size_opt.has_value()) {
    MS_LOG(INFO) << "Exit adaptive_max_pool2d pass due to tuple value cannot get.";
    return node;
  }

  auto output_size_val_get = output_size_opt.value();
  if (output_size_val_get.IsValueUnknown(kDim0) || output_size_val_get.IsValueUnknown(kDim1)) {
    MS_LOG(INFO) << "Exit adaptive_max_pool2d pass due to tuple value unknown.";
    return node;
  }

  auto output_size = output_size_val_get;

  if (output_size.size() != kShape2dDims) {
    MS_LOG(EXCEPTION) << "AdaptiveMaxPool2D's output_size shape should equal to 2.";
  }

  size_t w_index = input_shape.size() - kIndex1;
  size_t h_index = input_shape.size() - kIndex2;
  int64_t height = input_shape.at(h_index);
  int64_t width = input_shape.at(w_index);
  int64_t output_h = (output_size[kDim0] == -1) ? height : output_size[kDim0];
  int64_t output_w = (output_size[kDim1] == -1) ? width : output_size[kDim1];
  if ((output_h != -1 && output_h <= 0) || (output_w != -1 && output_w <= 0)) {
    MS_LOG(EXCEPTION) << "AdaptiveMaxPool2D's output_size value is invalid.";
  }

  if (AnfAlgo::GetOutputElementNum(adaptive_max_pool2d) > 1) {
    ShapeVector output_size_new = {output_h, output_w};
    auto output_size_new_node = kernel_graph->NewValueNode(MakeValue<std::vector<int64_t>>(output_size_new));
    output_size_new_node->set_abstract(output_size_node->abstract());

    auto input_node = adaptive_max_pool2d->input(kIndex1);
    auto adaptive_max_pool2d_new =
      func_graph->NewCNode({NewValueNode(prim::kPrimAdaptiveMaxPool2D), input_node, output_size_new_node});
    MS_EXCEPTION_IF_NULL(adaptive_max_pool2d_new);
    auto out_shape1 = AnfAlgo::GetOutputDetailShape(adaptive_max_pool2d, kIndex0);
    auto out_shape2 = AnfAlgo::GetOutputDetailShape(adaptive_max_pool2d, kIndex1);
    auto out_dtype1 = common::AnfAlgo::GetOutputInferDataType(adaptive_max_pool2d, kIndex0);
    auto out_dtype2 = common::AnfAlgo::GetOutputInferDataType(adaptive_max_pool2d, kIndex1);
    adaptive_max_pool2d_new->set_scope(adaptive_max_pool2d->scope());

    common::AnfAlgo::SetOutputTypeAndDetailShape({out_dtype1, out_dtype2}, {out_shape1, out_shape2},
                                                 adaptive_max_pool2d_new.get());
    return adaptive_max_pool2d_new;
  }

  if (height % output_h != 0 || width % output_w != 0) {
    auto types = {common::AnfAlgo::GetOutputInferDataType(adaptive_max_pool2d, 0), kNumberTypeInt64};
    auto shapes = {AnfAlgo::GetOutputDetailShape(adaptive_max_pool2d, 0),
                   AnfAlgo::GetOutputDetailShape(adaptive_max_pool2d, 0)};
    common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, adaptive_max_pool2d.get());
    std::vector<AnfNodePtr> multi_outputs;
    CreateMultipleOutputsOfAnfNode(func_graph, adaptive_max_pool2d, kAdaptiveMaxpool2DOutputNumber, &multi_outputs);
    return multi_outputs[kIndex0];
  }
  auto height_attr = ComputeKernelAttr(height, output_h);
  auto width_attr = ComputeKernelAttr(width, output_w);
  if (height_attr[kIndex0] == -1 || width_attr[kIndex0] == -1) {
    MS_LOG_WITH_NODE(EXCEPTION, node) << "Current AdaptiveMaxPool2D not support this scene! node:"
                                      << node->DebugString();
  }

  std::vector<AnfNodePtr> pooling_inputs = {NewValueNode(std::make_shared<Primitive>(kPoolingOpName))};
  (void)pooling_inputs.insert(pooling_inputs.end(), adaptive_max_pool2d->inputs().begin() + 1,
                              adaptive_max_pool2d->inputs().end());
  auto pooling = NewCNode(pooling_inputs, kernel_graph);
  auto types = {common::AnfAlgo::GetOutputInferDataType(adaptive_max_pool2d, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(adaptive_max_pool2d, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, pooling.get());
  pooling->set_scope(adaptive_max_pool2d->scope());
  SetNodeAttr(pooling, height_attr, width_attr);
  return pooling;
}
}  // namespace opt
}  // namespace mindspore
