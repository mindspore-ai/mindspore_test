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
#include "infer/ops_func_impl/quant_linear_sparse.h"
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "abstract/dshape.h"
#include "utils/shape_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
constexpr size_t kQuantLinearSparseInputX = 0;
constexpr size_t kQuantLinearSparseInputWeight = 1;
constexpr size_t kQuantLinearSparseInputDeqScale = 2;
constexpr size_t kQuantLinearSparseInputCompressIdx = 3;
constexpr size_t kQuantLinearSparseInputBias = 4;

BaseShapePtr QuantLinearSparseFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const std::vector<AbstractBasePtr> &input_args) const {
  auto x_shape = input_args[kQuantLinearSparseInputX]->GetShape();
  auto x_shape_vector = x_shape->GetShapeVector();
  auto x_rank = x_shape_vector.size();
  if (x_rank != kDim2) {
    MS_LOG(EXCEPTION) << "For 'QuantLinearSparse', x's rank should be 2, but got " << x_rank;
  }

  auto m = x_shape_vector[kIndex0];
  auto k = x_shape_vector[kIndex1];

  auto deqScale_shape = input_args[kQuantLinearSparseInputDeqScale]->GetShape();
  auto deqScale_shape_vector = deqScale_shape->GetShapeVector();
  auto deqScale_rank = deqScale_shape_vector.size();
  if (deqScale_rank != kDim1) {
    MS_LOG(EXCEPTION) << "For 'QuantLinearSparse', deqScale's rank should be 1, but got " << deqScale_rank;
  }
  auto n = deqScale_shape_vector[deqScale_rank - 1];

  if (!input_args[kQuantLinearSparseInputBias]->GetType()->isa<TypeNone>()) {
    auto bias_shape = input_args[4]->GetShape();
    auto bias_shape_vector = bias_shape->GetShapeVector();
    auto bias_rank = bias_shape_vector.size();

    if (bias_rank != 1 || bias_shape_vector[bias_rank - 1] != n) {
      MS_LOG(EXCEPTION) << "For 'QuantLinearSparse', bias's shape should be the same as deqScale's, but got "
                        << bias_shape_vector;
    }
  }

  auto weight_shape = input_args[kQuantLinearSparseInputWeight]->GetShape();
  auto weight_shape_vector = weight_shape->GetShapeVector();
  auto weight_rank = weight_shape_vector.size();
  if (weight_rank != 1) {
    MS_LOG(EXCEPTION) << "For 'QuantLinearSparse', weight's rank should be 1, but got " << weight_rank;
  }
  auto c = weight_shape_vector[0];

  if (c == 0 || c > k * n) {
    MS_LOG(EXCEPTION) << "For 'QuantLinearSparse', weight's shape should be [c] and c should be in [1,k*n], but got "
                      << weight_shape_vector;
  }
  auto compressIdx_shape = input_args[kQuantLinearSparseInputCompressIdx]->GetShape();
  auto compressIdx_shape_vector = compressIdx_shape->GetShapeVector();
  auto compressIdx_rank = compressIdx_shape_vector.size();
  if (compressIdx_rank != 1) {
    MS_LOG(EXCEPTION) << "For 'QuantLinearSparse', compressIdx’s rank should be 1, but got " << compressIdx_rank;
  }

  std::vector<int64_t> output_shape = {m, n};
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr QuantLinearSparseFuncImpl::InferType(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) const {
  std::map<std::string, TypePtr> types;
  MS_EXCEPTION_IF_NULL(input_args[kQuantLinearSparseInputX]);
  TypePtr x_type = input_args[kQuantLinearSparseInputX]->GetType();
  MS_EXCEPTION_IF_NULL(input_args[kQuantLinearSparseInputWeight]);
  TypePtr weight_type = input_args[kQuantLinearSparseInputWeight]->GetType();
  MS_EXCEPTION_IF_NULL(input_args[kQuantLinearSparseInputCompressIdx]);
  TypePtr compressIdx_type = input_args[kQuantLinearSparseInputCompressIdx]->GetType();
  (void)types.emplace("x", x_type);
  (void)types.emplace("weight", weight_type);
  (void)types.emplace("compressIdx", compressIdx_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt8}, primitive->name());

  MS_EXCEPTION_IF_NULL(input_args[kQuantLinearSparseInputBias]);
  if (!input_args[kQuantLinearSparseInputBias]->GetType()->isa<TypeNone>()) {
    TypePtr bias_type = input_args[kQuantLinearSparseInputBias]->GetType();
    types.clear();
    (void)types.emplace("bias", bias_type);
    (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt32}, primitive->name());
  }
  MS_EXCEPTION_IF_NULL(input_args[kQuantLinearSparseInputDeqScale]);
  TypePtr deqScale_type = input_args[kQuantLinearSparseInputDeqScale]->GetType();
  types.clear();
  (void)types.emplace("deqScale", deqScale_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt64, kUInt64}, primitive->name());

  return kFloat16;
}
}  // namespace mindspore::ops
