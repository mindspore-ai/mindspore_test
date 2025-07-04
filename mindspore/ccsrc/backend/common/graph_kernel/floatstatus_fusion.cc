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
#include "backend/common/graph_kernel/floatstatus_fusion.h"
#include <algorithm>
#include <memory>
#include <string>
#include <vector>
#include "include/backend/anf_runtime_algorithm.h"
#include "include/backend/optimizer/helper.h"
#include "mindspore/ops/op_def/array_ops.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_c.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_f.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_r.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_s.h"

namespace mindspore::graphkernel {
const BaseRef FloatStatusBaseFusion::DefinePattern() const {
  VectorRef is_finite = VectorRef({isfinite_prim_, input_});
  VectorRef reduce = VectorRef({prim::kPrimReduceAll, is_finite, axis_, keep_dims_});
  VectorRef cast = VectorRef({prim::kPrimCast, reduce, type_});
  VectorRef sub = VectorRef({prim::kPrimSub, s_, cast});
  return sub;
}

const BaseRef FloatStatusReshapeFusion::DefinePattern() const {
  return VectorRef({prim::kPrimReshape, FloatStatusBaseFusion::DefinePattern(), to_shape_});
}

const BaseRef CastFloatStatusBaseFusion::DefinePattern() const {
  VectorRef cast_tmp = VectorRef({prim::kPrimCast, input_, type_fp32_});
  VectorRef is_finite = VectorRef({isfinite_prim_, cast_tmp});
  VectorRef reduce = VectorRef({prim::kPrimReduceAll, is_finite, axis_, keep_dims_});
  VectorRef cast = VectorRef({prim::kPrimCast, reduce, type_});
  VectorRef sub = VectorRef({prim::kPrimSub, s_, cast});
  return sub;
}

const BaseRef CastFloatStatusReshapeFusion::DefinePattern() const {
  return VectorRef({prim::kPrimReshape, CastFloatStatusBaseFusion::DefinePattern(), to_shape_});
}

const AnfNodePtr FloatStatusBaseFusion::Process(const FuncGraphPtr &func_graph, const AnfNodePtr &node,
                                                const EquivPtr &equiv) const {
  auto mng = func_graph->manager();
  if (mng == nullptr) {
    mng = Manage(func_graph, false);
    func_graph->set_manager(mng);
  }
  auto &users = mng->node_users()[node];
  if (std::any_of(users.begin(), users.end(), [](const CNodeIndexPair &index_pair) {
        return !IsPrimitiveCNode(index_pair.first, prim::kPrimAddN);
      })) {
    return nullptr;
  }
  auto input_node = opt::GetAnfNodeByVar(equiv, input_);
  auto axis_node = opt::GetAnfNodeByVar(equiv, axis_);
  auto isfinite_node = opt::GetAnfNodeByVar(equiv, isfinite_prim_);
  if (!axis_node->isa<ValueNode>()) {
    return nullptr;
  }
  auto axis_value = axis_node->cast<ValueNodePtr>()->value();
  ShapeVector axis_vector;
  if (axis_value->isa<tensor::Tensor>()) {
    axis_vector = TensorValueToVector<int64_t>(axis_value->cast<tensor::TensorPtr>());
  } else {
    axis_vector = GetValue<ShapeVector>(axis_node->cast<ValueNodePtr>()->value());
  }
  auto input_vector = AnfAlgo::GetInputDeviceShape(isfinite_node, 0);
  if (!axis_vector.empty()) {
    std::sort(axis_vector.begin(), axis_vector.end());
    auto unique_size = static_cast<size_t>(std::unique(axis_vector.begin(), axis_vector.end()) - axis_vector.begin());
    if (unique_size != input_vector.size()) {
      return nullptr;
    }
  }
  auto input_type = AnfAlgo::GetOutputDeviceDataType(input_node, 0);
  auto output_type = AnfAlgo::GetOutputDeviceDataType(node, 0);
  if (input_type != output_type || output_type != kNumberTypeFloat32) {
    return nullptr;
  }
  auto s_node = opt::GetAnfNodeByVar(equiv, s_);
  if (!s_node->isa<ValueNode>()) {
    return nullptr;
  }
  auto value = s_node->cast<ValueNodePtr>()->value();
  MS_EXCEPTION_IF_NULL(value);
  auto s_value = TensorValueToVector<float>(value->cast<tensor::TensorPtr>())[0];
  if (s_value != 1) {
    return nullptr;
  }
  auto new_node = func_graph->NewCNode(prim::kPrimFloatStatus, {input_node});
  new_node->set_abstract(node->abstract());
  kernel::KernelBuildInfo::KernelBuildInfoBuilder info_builder;
  info_builder.SetInputsFormat({kOpFormat_DEFAULT});
  info_builder.SetInputsDeviceType({output_type});
  info_builder.SetOutputsFormat({kOpFormat_DEFAULT});
  info_builder.SetOutputsDeviceType({output_type});
  info_builder.SetKernelType(UNKNOWN_KERNEL_TYPE);
  info_builder.SetOutputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  info_builder.SetInputsKernelObjectType({kernel::KernelObjectType::TENSOR});
  AnfAlgo::SetSelectKernelBuildInfo(info_builder.Build(), new_node.get());
  return new_node;
}
}  // namespace mindspore::graphkernel
