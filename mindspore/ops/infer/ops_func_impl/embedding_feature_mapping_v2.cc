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

#include "infer/ops_func_impl/embedding_feature_mapping_v2.h"

#include <utility>
#include <memory>

#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr EmbeddingFeatureMappingV2FuncImpl::InferShape(const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) const {
  auto feature_id_shape = input_args[kIndex1]->GetShape()->GetShapeVector();
  MS_CHECK_VALUE(feature_id_shape.size() != 0,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("feature_id's rank", SizeToLong(feature_id_shape.size()),
                                                             kNotEqual, int64_t(0), primitive));
  return std::make_shared<abstract::TensorShape>(std::move(feature_id_shape));
}

TypePtr EmbeddingFeatureMappingV2FuncImpl::InferType(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  const auto &prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("feature_id", input_args[1]->GetType(), {kInt64}, prim_name);
  return std::make_shared<TensorType>(kInt32);
}
}  // namespace ops
}  // namespace mindspore
