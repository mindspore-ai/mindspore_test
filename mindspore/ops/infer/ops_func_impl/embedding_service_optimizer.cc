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

#include "infer/ops_func_impl/embedding_service_optimizer.h"

#include <memory>
#include <vector>

#include "utils/check_convert_utils.h"
#include "ops_utils/op_utils.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
BaseShapePtr EmbeddingServiceOptimizerFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) const {
  CheckInputShapes(primitive, input_args);
  return std::make_shared<abstract::TensorShape>(ShapeVector{});
}

TypePtr EmbeddingServiceOptimizerFuncImpl::InferType(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  CheckInputTypes(primitive, input_args);
  return std::make_shared<TensorType>(kInt32);
}

int32_t EmbeddingServiceOptimizerFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                           const std::vector<AbstractBasePtr> &input_args) const {
  const auto &embedding_dims_opt = GetArrayValue<int64_t>(input_args[embedding_dim_index_]);
  if (MS_UNLIKELY(embedding_dims_opt.has_value())) {
    return OP_CHECK_SUCCESS;
  }

  auto embedding_dims = embedding_dims_opt.value();
  for (size_t i = 0; i < embedding_dims.size(); ++i) {
    if (MS_UNLIKELY(embedding_dims.IsValueUnknown(i))) {
      continue;
    }
    MS_CHECK_VALUE(embedding_dims[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                            "embedding_dim", embedding_dims[i], kGreaterThan, int64_t(0), primitive));
  }
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
