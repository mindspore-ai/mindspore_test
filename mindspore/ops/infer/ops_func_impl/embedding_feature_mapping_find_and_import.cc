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

#include "infer/ops_func_impl/embedding_feature_mapping_find.h"
#include "infer/ops_func_impl/embedding_feature_mapping_import.h"
#include "abstract/dshape.h"
#include "ops_utils/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/shape_utils.h"

namespace mindspore {
namespace ops {
namespace {
size_t FetchEmbeddingFeatureMappingTableNum(const PrimitivePtr &primitive, const AbstractBasePtr &table_name_arg) {
  const auto &table_name_shape = table_name_arg->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamic(table_name_shape))) {
    MS_EXCEPTION(RuntimeError) << "For " << primitive->name()
                               << ", table_name_shape should not be dynamic, which has not been supported.";
  }
  auto table_num = SizeOf(table_name_shape);
  if (table_num != kIndex1) {
    MS_EXCEPTION(ValueError) << "For " << primitive->name()
                             << ", the cases have not been supported where table_num is not 1, but got " << table_num;
  }
  return table_num;
}

int32_t CheckEmbeddingFeatureMappingNum(const PrimitivePtr &primitive, const AbstractBasePtr &num_arg,
                                        int64_t table_num) {
  auto num_opt = GetScalarValue<int64_t>(num_arg->GetValue());
  if (MS_UNLIKELY(!num_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  auto num = num_opt.value();
  MS_CHECK_VALUE(table_num == num,
                 CheckAndConvertUtils::FormatCheckIntegerMsg("num", num, kEqual, table_num, primitive));
  return OP_CHECK_SUCCESS;
}

void EmbeddingFeatureMappingInsertMultiDynamicShapes(abstract::BaseShapePtrList *const shapes, size_t num) {
  for (size_t i = 0; i < num; i++) {
    shapes->emplace_back(std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeDimAny}));
  }
}

std::tuple<abstract::BaseShapePtrList, std::optional<ArrayValue<int64_t>>, size_t>
EmbeddingFeatureMappingInferFeatureIdAndOffsetIdShapes(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args,
                                                       size_t table_name_idx, size_t feature_size_idx) {
  auto table_num = FetchEmbeddingFeatureMappingTableNum(primitive, input_args[table_name_idx]);
  auto feature_size_opt = GetArrayValue<int64_t>(input_args[feature_size_idx]);
  abstract::BaseShapePtrList shapes;
  // feature_id shapes
  if (MS_LIKELY(feature_size_opt.has_value())) {
    const auto &feature_size = feature_size_opt.value();
    MS_CHECK_VALUE(feature_size.size() == table_num,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("num of feature_size", SizeToLong(feature_size.size()),
                                                               kEqual, SizeToLong(table_num), primitive));
    for (size_t i = 0; i < table_num; i++) {
      if (feature_size.IsValueUnknown(i)) {
        shapes.emplace_back(std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeDimAny}));
      } else {
        auto feature_size_i = feature_size[i];
        MS_CHECK_VALUE(feature_size_i > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("feature_size", feature_size_i,
                                                                                       kGreaterThan, 0, primitive));
        shapes.emplace_back(std::make_shared<abstract::TensorShape>(ShapeVector{feature_size_i}));
      }
    }
  } else {
    EmbeddingFeatureMappingInsertMultiDynamicShapes(&shapes, table_num);
  }
  // offset_id shapes
  std::transform(shapes.begin(), shapes.begin() + table_num, std::back_inserter(shapes),
                 [](const BaseShapePtr &shape_ptr) { return shape_ptr->Clone(); });

  return std::make_tuple(std::move(shapes), feature_size_opt, table_num);
}

std::tuple<TypePtrList, size_t> EmbeddingFeatureMappingInferFeatureIdAndOffsetIdTypes(
  const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args, size_t table_name_idx) {
  const auto &table_name_shape = input_args[table_name_idx]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamic(table_name_shape))) {
    MS_EXCEPTION(RuntimeError) << "For " << primitive->name()
                               << ", table_name_shape should not be dynamic, which has not been supported.";
  }
  auto table_num = SizeOf(table_name_shape);
  std::vector<TypePtr> out_types(table_num, kInt64);
  out_types.insert(out_types.end(), table_num, kInt32);
  return std::make_tuple(std::move(out_types), table_num);
}
}  // namespace

BaseShapePtr EmbeddingFeatureMappingFindFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                             const std::vector<AbstractBasePtr> &input_args) const {
  abstract::BaseShapePtrList shapes;
  size_t table_num;
  std::tie(shapes, std::ignore, table_num) =
    EmbeddingFeatureMappingInferFeatureIdAndOffsetIdShapes(primitive, input_args, table_name_idx_, feature_size_idx_);
  (void)CheckEmbeddingFeatureMappingNum(primitive, input_args[num_idx_], SizeToLong(table_num));
  return std::make_shared<abstract::TupleShape>(std::move(shapes));
}

TypePtr EmbeddingFeatureMappingFindFuncImpl::InferType(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) const {
  TypePtrList out_types;
  std::tie(out_types, std::ignore) =
    EmbeddingFeatureMappingInferFeatureIdAndOffsetIdTypes(primitive, input_args, table_name_idx_);
  return std::make_shared<Tuple>(std::move(out_types));
}

BaseShapePtr EmbeddingFeatureMappingImportFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto [shapes, feature_size_opt, table_num] =
    EmbeddingFeatureMappingInferFeatureIdAndOffsetIdShapes(primitive, input_args, table_name_idx_, feature_size_idx_);
  // value shapes
  auto embedding_dim_opt = GetArrayValue<int64_t>(input_args[embedding_dim_idx_]);
  if (MS_LIKELY(feature_size_opt.has_value() && embedding_dim_opt.has_value())) {
    const auto &feature_size = feature_size_opt.value();
    const auto &embedding_dim = embedding_dim_opt.value();
    MS_CHECK_VALUE(embedding_dim.size() == table_num,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("num of embedding_dim", embedding_dim.size(), kEqual,
                                                               table_num, primitive));
    for (size_t i = 0; i < table_num; i++) {
      if (MS_UNLIKELY(embedding_dim.IsValueUnknown(i) || feature_size.IsValueUnknown(i))) {
        shapes.emplace_back(std::make_shared<abstract::TensorShape>(ShapeVector{abstract::Shape::kShapeDimAny}));
        continue;
      }
      shapes.emplace_back(std::make_shared<abstract::TensorShape>(ShapeVector{embedding_dim[i] * feature_size[i]}));
    }
  } else {
    EmbeddingFeatureMappingInsertMultiDynamicShapes(&shapes, table_num);
  }

  return std::make_shared<abstract::TupleShape>(std::move(shapes));
}

TypePtr EmbeddingFeatureMappingImportFuncImpl::InferType(const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) const {
  auto [out_types, table_num] =
    EmbeddingFeatureMappingInferFeatureIdAndOffsetIdTypes(primitive, input_args, table_name_idx_);
  out_types.insert(out_types.end(), table_num, kFloat32);
  return std::make_shared<Tuple>(std::move(out_types));
}

int32_t EmbeddingFeatureMappingImportFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto embedding_dim_opt = GetArrayValue<int64_t>(input_args[embedding_dim_idx_]);
  if (MS_UNLIKELY(!embedding_dim_opt.has_value())) {
    return OP_CHECK_RETRY;
  }
  auto embedding_dim = embedding_dim_opt.value();
  for (size_t i = 0; i < embedding_dim.size(); i++) {
    if (MS_UNLIKELY(embedding_dim.IsValueUnknown(i))) {
      return OP_CHECK_RETRY;
    }
    MS_CHECK_VALUE(embedding_dim[i] > 0, CheckAndConvertUtils::FormatCheckIntegerMsg("embedding_dim", embedding_dim[i],
                                                                                     kGreaterThan, 0, primitive));
  }
  auto table_num = SizeToLong(embedding_dim.size());
  return CheckEmbeddingFeatureMappingNum(primitive, input_args[num_idx_], table_num);
}
}  // namespace ops
}  // namespace mindspore
