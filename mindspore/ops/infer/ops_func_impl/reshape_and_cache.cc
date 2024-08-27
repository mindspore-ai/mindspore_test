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

#include "infer/ops_func_impl/reshape_and_cache.h"
#include <utility>
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr ReshapeAndCacheFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                 const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();
  auto ordinary_input_num = CheckAndConvertUtils::GetRemoveUMonadAbsNum(input_args);
  (void)CheckAndConvertUtils::CheckInteger("inputs num", SizeToLong(ordinary_input_num), kEqual,
                                           kReshapeAndCacheInputsNum, op_name);
  auto key_shape_ptr = input_args[kReshapeAndCacheInputKeyIndex]->GetShape();
  if (MS_UNLIKELY(IsDynamicRank(key_shape_ptr->GetShapeVector()))) {
    ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }

  auto slot_mapping_shape_ptr = input_args[kReshapeAndCacheInputSlotMappingIndex]->GetShape();
  if (MS_UNLIKELY(IsDynamicRank(slot_mapping_shape_ptr->GetShapeVector()))) {
    ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }

  if (!IsDynamicShape(key_shape_ptr->GetShapeVector()) && !IsDynamicShape(slot_mapping_shape_ptr->GetShapeVector())) {
    auto slot_mapping_token_size = slot_mapping_shape_ptr->GetShapeVector()[kInputIndex0];
    auto key_token_size = key_shape_ptr->GetShapeVector()[kInputIndex0] * key_shape_ptr->GetShapeVector()[kInputIndex1];
    if (slot_mapping_token_size != key_token_size) {
      MS_LOG(EXCEPTION) << "The num_tokens of slot mapping and key must be the same, but got slot mapping num_tokens: "
                        << slot_mapping_token_size << ", key num_tokens: " << key_token_size;
    }
  }
  auto shape_element = key_shape_ptr->cast<abstract::ShapePtr>();
  return shape_element;  // output shape
}

TypePtr ReshapeAndCacheFuncImpl::InferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) const {
  const std::set valid_types = {kFloat16, kBFloat16, kInt8, kUInt8};
  auto op_name = primitive->name();
  std::map<std::string, TypePtr> types;

  (void)types.emplace("key", input_args[kReshapeAndCacheInputKeyIndex]->GetType());
  (void)types.emplace("value", input_args[kReshapeAndCacheInputValueIndex]->GetType());
  (void)types.emplace("key_cache", input_args[kReshapeAndCacheInputKeyCacheIndex]->GetType());
  (void)types.emplace("value_cache", input_args[kReshapeAndCacheInputValueCacheIndex]->GetType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);

  return type;  // output type
}
}  // namespace ops
}  // namespace mindspore
