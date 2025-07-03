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

#include "infer/ops_func_impl/embedding_dense_backward.h"

#include <memory>

#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/log_adapter.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
BaseShapePtr EmbeddingDenseBackwardFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                        const std::vector<AbstractBasePtr> &input_args) const {
  auto num_weights_opt = GetScalarValue<int64_t>(input_args[kIndex2]->GetValue());
  auto num_embedding = num_weights_opt.value_or(abstract::Shape::kShapeDimAny);

  const auto &grad_shape = input_args[kIndex0]->GetShape()->GetShapeVector();
  MS_EXCEPTION_IF_ZERO("size of grad_shape", grad_shape.size());
  auto embedding_dim = IsDynamicRank(grad_shape) ? abstract::Shape::kShapeDimAny : grad_shape.back();

  return std::make_shared<abstract::Shape>(ShapeVector{num_embedding, embedding_dim});
}

TypePtr EmbeddingDenseBackwardFuncImpl::InferType(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) const {
  return input_args[kIndex0]->GetType();
}

ShapeArray EmbeddingDenseBackwardFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                      const ValuePtrList &input_values) const {
  auto grad_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(grad_tensor);
  const auto &grad_shape = grad_tensor->shape();

  auto num_weights_opt = GetScalarValue<int64_t>(input_values[kIndex2]);
  MS_ASSERT(num_weights_opt.has_value());
  auto num_embedding = num_weights_opt.value();

  MS_EXCEPTION_IF_ZERO("size of grad_shape", grad_shape.size());
  auto embedding_dim = grad_shape.back();

  return {ShapeVector{num_embedding, embedding_dim}};
}

TypePtrList EmbeddingDenseBackwardFuncImpl::InferType(const PrimitivePtr &primitive,
                                                      const ValuePtrList &input_values) const {
  auto grad_tensor = input_values[kIndex0]->cast<tensor::TensorPtr>();
  MS_EXCEPTION_IF_NULL(grad_tensor);
  return {grad_tensor->Dtype()};
}
}  // namespace ops
}  // namespace mindspore
