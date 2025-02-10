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

#include "infer/ops_func_impl/group_topk.h"

#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/helper.h"

namespace mindspore {
namespace ops {
BaseShapePtr GroupTopkFuncImpl::InferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  auto ordinary_input_num = CheckAndConvertUtils::GetRemoveUMonadAbsNum(input_args);
  (void)CheckAndConvertUtils::CheckInteger("inputs num", SizeToLong(ordinary_input_num), kEqual, kGroupTopkInputsNum,
                                           op_name);
  auto token_shape_ptr = input_args[kGroupTopkTokenIndex]->GetShape();
  if (MS_UNLIKELY(IsDynamicRank(token_shape_ptr->GetShapeVector()))) {
    ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }

  auto idx_arr_shape_ptr = input_args[kGroupTopkIdxArrIndex]->GetShape();
  if (MS_UNLIKELY(IsDynamicRank(idx_arr_shape_ptr->GetShapeVector()))) {
    ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }
  auto shape_element = token_shape_ptr->cast<abstract::ShapePtr>();
  return shape_element;  // output shape
}

TypePtr GroupTopkFuncImpl::InferType(const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  auto token_type = input_args[kGroupTopkTokenIndex]->GetType();
  auto idx_arr_type = input_args[kGroupTopkIdxArrIndex]->GetType();
  auto token_element_type = kNumberTypeBegin;
  auto idx_arr_element_type = kNumberTypeBegin;
  if (token_type->isa<TensorType>()) {
    auto tensor_type = token_type->cast<TensorTypePtr>();
    auto element = tensor_type->element();
    token_element_type = element->type_id();
  }
  if (idx_arr_type->isa<TensorType>()) {
    auto tensor_type = idx_arr_type->cast<TensorTypePtr>();
    auto element = tensor_type->element();
    idx_arr_element_type = element->type_id();
  }
  if ((token_element_type != kNumberTypeFloat16 && token_element_type != kNumberTypeBFloat16) ||
      idx_arr_element_type != kNumberTypeInt32) {
    MS_EXCEPTION(TypeError) << "The primitive[" << op_name << "]'s input arguments[token, idxArr], invalid type list: {"
                            << TypeIdToString(token_element_type) << "," << TypeIdToString(idx_arr_element_type) << "}";
  }

  return token_type;  // output type
}
}  // namespace ops
}  // namespace mindspore
