/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "plugin/device/gpu/optimizer/apply_momentum_scale_fusion.h"

#include <vector>
#include <string>

#include "mindspore/ops/op_def/nn_optimizer_ops.h"
#include "mindspore/ops/op_def/math_ops.h"
#include "ir/primitive.h"
#include "include/common/utils/utils.h"
#include "include/backend/optimizer/helper.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_a.h"
#include "mindspore/ops/op_def/auto_generate/gen_ops_primitive_m.h"

namespace mindspore {
namespace opt {
bool ApplyMomentumScaleFusion::IsScalar(const BaseRef &n) {
  if (utils::isa<AnfNodePtr>(n)) {
    AnfNodePtr in = utils::cast<AnfNodePtr>(n);
    MS_EXCEPTION_IF_NULL(in);
    auto shape_ptr = in->Shape();
    MS_EXCEPTION_IF_NULL(shape_ptr);
    auto shape = shape_ptr->cast<abstract::ShapePtr>();
    MS_EXCEPTION_IF_NULL(shape);
    if (shape->shape().size() != 0) {
      return false;
    }
    auto dtype = in->Type();
    if (dtype->type_id() != kObjectTypeTensorType) {
      return false;
    }
    auto type_ptr = dyn_cast<TensorType>(dtype);
    MS_EXCEPTION_IF_NULL(type_ptr);
    auto element = type_ptr->element();
    MS_EXCEPTION_IF_NULL(element);
    auto element_type = element->type_id();
    if (element_type != kNumberTypeFloat32) {
      return false;
    }
    return true;
  }
  return false;
}

const BaseRef ApplyMomentumScaleFusion::DefinePattern() const {
  VectorRef scale = VectorRef({prim::kPrimMul, gradient_, scale_});
  VectorRef apply_momentum =
    VectorRef({prim::kPrimApplyMomentum, variable_, accumulation_, learning_rate_, scale, momentum_, monad_state_});
  return apply_momentum;
}

const AnfNodePtr ApplyMomentumScaleFusion::Process(const FuncGraphPtr &graph, const AnfNodePtr &node,
                                                   const EquivPtr &equiv) const {
  MS_EXCEPTION_IF_NULL(graph);
  MS_EXCEPTION_IF_NULL(node);
  MS_EXCEPTION_IF_NULL(equiv);
  auto scale = utils::cast<AnfNodePtr>((*equiv)[scale_]);
  auto variable = utils::cast<AnfNodePtr>((*equiv)[variable_]);
  auto accumulation = utils::cast<AnfNodePtr>((*equiv)[accumulation_]);
  auto learning_rate = utils::cast<AnfNodePtr>((*equiv)[learning_rate_]);
  auto gradient = utils::cast<AnfNodePtr>((*equiv)[gradient_]);
  auto momentum = utils::cast<AnfNodePtr>((*equiv)[momentum_]);
  auto monad_state = utils::cast<AnfNodePtr>((*equiv)[monad_state_]);
  MS_EXCEPTION_IF_NULL(scale);
  MS_EXCEPTION_IF_NULL(variable);
  MS_EXCEPTION_IF_NULL(accumulation);
  MS_EXCEPTION_IF_NULL(learning_rate);
  MS_EXCEPTION_IF_NULL(gradient);
  MS_EXCEPTION_IF_NULL(momentum);
  MS_EXCEPTION_IF_NULL(monad_state);

  auto prim = std::make_shared<Primitive>(kFusedScaleApplyMomentumOpName);
  MS_EXCEPTION_IF_NULL(prim);
  std::vector<AnfNodePtr> inputs = {NewValueNode(prim), scale,    variable, accumulation,
                                    learning_rate,      gradient, momentum, monad_state};
  auto replace_node = graph->NewCNode(inputs);
  MS_EXCEPTION_IF_NULL(replace_node);
  auto types = {common::AnfAlgo::GetOutputInferDataType(node, 0)};
  auto shapes = {AnfAlgo::GetOutputDetailShape(node, 0)};
  common::AnfAlgo::SetOutputTypeAndDetailShape(types, shapes, replace_node.get());
  replace_node->set_scope(node->scope());
  return replace_node;
}
}  // namespace opt
}  // namespace mindspore
