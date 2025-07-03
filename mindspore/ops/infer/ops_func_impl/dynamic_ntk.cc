/**
 * Copyright 2025 Huawei Technologies Co., Ltd
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
#include "infer/ops_func_impl/dynamic_ntk.h"
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "abstract/dshape.h"
#include "utils/shape_utils.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore::ops {
constexpr size_t kDynamicNTKInputPositionIds = 0;
constexpr size_t kDynamicNTKInputInvFreq = 1;
constexpr size_t kDynamicNTKInputSeqLens = 2;
constexpr size_t kDynamicNTKInputOutType = 3;

BaseShapePtr DynamicNTKFuncImpl::InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) const {
  auto position_ids_shape = input_args[kDynamicNTKInputPositionIds]->GetShape();
  const auto &position_ids_vector = position_ids_shape->GetShapeVector();
  auto position_ids_rank = position_ids_vector.size();
  if (position_ids_rank != kDim1) {
    MS_LOG(EXCEPTION) << "For 'DynamicNTK', position_ids's rank should be 1, but got " << position_ids_rank;
  }
  auto ntokens = position_ids_vector[kDim0];

  auto inv_freq_shape = input_args[kDynamicNTKInputInvFreq]->GetShape();
  const auto &inv_freq_vector = inv_freq_shape->GetShapeVector();
  auto inv_freq_rank = inv_freq_vector.size();
  if (inv_freq_rank != kDim2) {
    MS_LOG(EXCEPTION) << "For 'DynamicNTK', inv_freq's rank should be 2, but got " << inv_freq_rank;
  }
  auto head_dim = inv_freq_vector[kDim1] * SizeToLong(kDim2);  // headDim is twice the second dimension of inv_freq

  auto seqlens_shape = input_args[kDynamicNTKInputSeqLens]->GetShape();
  const auto &seqlens_vector = seqlens_shape->GetShapeVector();
  auto seqlens_rank = seqlens_vector.size();
  if (seqlens_rank != kDim1) {
    MS_LOG(EXCEPTION) << "For 'DynamicNTK', seqlens's rank should be 1, but got " << seqlens_rank;
  }

  std::vector<int64_t> output_shape = {ntokens, head_dim};
  std::vector<BaseShapePtr> shapes_list;
  shapes_list.push_back(std::make_shared<abstract::Shape>(output_shape));
  shapes_list.push_back(std::make_shared<abstract::Shape>(output_shape));
  return std::make_shared<abstract::TupleShape>(shapes_list);
}

TypePtr DynamicNTKFuncImpl::InferType(const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) const {
  std::map<std::string, TypePtr> types;
  TypePtr position_ids_type = input_args[kDynamicNTKInputPositionIds]->GetType();
  types.emplace("position_ids", position_ids_type);
  CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt32}, primitive->name());

  types.clear();
  TypePtr inv_freq_type = input_args[kDynamicNTKInputInvFreq]->GetType();
  types.emplace("inv_freq", inv_freq_type);
  CheckAndConvertUtils::CheckTensorTypeSame(types, {kFloat32}, primitive->name());

  types.clear();
  TypePtr seqlens_type = input_args[kDynamicNTKInputSeqLens]->GetType();
  types.emplace("seqlens", seqlens_type);
  CheckAndConvertUtils::CheckTensorTypeSame(types, {kInt32}, primitive->name());

  auto out_type_opt = GetScalarValue<int64_t>(input_args[kDynamicNTKInputOutType]->GetValue());
  if (!out_type_opt.has_value()) {
    MS_EXCEPTION(ValueError) << "For op 'DynamicNTK', out_type can't be dynamic";
  }
  auto out_type = static_cast<TypeId>(out_type_opt.value());
  auto out_dtype = kFloat16;
  if (out_type == TypeId::kNumberTypeFloat16) {
    out_dtype = kFloat16;
  } else if (out_type == TypeId::kNumberTypeBFloat16) {
    out_dtype = kBFloat16;
  } else if (out_type == TypeId::kNumberTypeFloat32) {
    out_dtype = kFloat32;
  } else {
    MS_LOG(EXCEPTION) << "For 'DynamicNTK', out_type should be float16, bfloat16 or float32, but got " << out_type;
  }

  std::vector<TypePtr> types_list = {out_dtype, out_dtype};
  return std::make_shared<Tuple>(types_list);
}
}  // namespace mindspore::ops
