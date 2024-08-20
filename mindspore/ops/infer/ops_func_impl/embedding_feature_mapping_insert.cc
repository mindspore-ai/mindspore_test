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
#include "infer/ops_func_impl/embedding_feature_mapping_insert.h"

#include <string>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "ops_utils/op_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
void EmbeddingFeatureMappingInsertCheckRank(const PrimitivePtr &primitive, const std::vector<int64_t> &shape,
                                            const std::string &arg_name) {
  MS_CHECK_VALUE(shape.size() != 0,
                 CheckAndConvertUtils::FormatCheckIntegerMsg(arg_name + "'s rank", SizeToLong(shape.size()), kNotEqual,
                                                             int64_t(0), primitive));
}

void EmbeddingFeatureMappingCheckFeatureAndOffsetIdShape(const PrimitivePtr &primitive,
                                                         const ShapeVector &feature_id_shape,
                                                         const ShapeVector &offset_id_shape,
                                                         std::vector<int64_t> *const feature_size) {
  EmbeddingFeatureMappingInsertCheckRank(primitive, feature_id_shape, "feature_id");
  EmbeddingFeatureMappingInsertCheckRank(primitive, offset_id_shape, "offset_id");
  if (MS_LIKELY(!(IsDynamic(feature_id_shape) || IsDynamic(offset_id_shape)))) {
    if (MS_UNLIKELY(feature_id_shape != offset_id_shape)) {
      MS_EXCEPTION(RuntimeError) << "For " << primitive->name()
                                 << ", the shapes of feature_id and offset_id should be the same, but got "
                                 << feature_id_shape << " and " << offset_id_shape;
    }
    feature_size->emplace_back(SizeToLong(SizeOf(feature_id_shape)));
  }
}

std::vector<int64_t> EmbeddingFeatureMappingCheckFeatureAndOffsetId(const PrimitivePtr &primitive,
                                                                    const std::vector<AbstractBasePtr> &input_args,
                                                                    size_t start_idx, size_t table_num) {
  std::vector<int64_t> feature_size;
  if (MS_LIKELY(!CheckAndConvertUtils::IsTensor(input_args.at(start_idx)) &&
                !CheckAndConvertUtils::IsTensor(input_args.at(start_idx + kIndex1)))) {
    auto feature_base_shape = input_args[start_idx]->GetShape();
    auto offset_base_shape = input_args[start_idx + kIndex1]->GetShape();
    if (MS_UNLIKELY(feature_base_shape->isa<abstract::DynamicSequenceShape>() ||
                    offset_base_shape->isa<abstract::DynamicSequenceShape>())) {
      MS_EXCEPTION(RuntimeError) << "For " << primitive->name()
                                 << ", all inputs should not be dynamic sequence which were not supported.";
    }
    auto feature_id_shapes = feature_base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(feature_id_shapes);
    auto offset_id_shapes = offset_base_shape->cast<abstract::TupleShapePtr>();
    MS_EXCEPTION_IF_NULL(offset_id_shapes);
    assert(feature_id_shapes->size() == offset_id_shapes->size());
    assert(feature_id_shapes->size() == table_num);
    for (size_t i = 0; i < table_num; i++) {
      const auto &feature_id_shape = (*feature_id_shapes)[i]->GetShapeVector();
      const auto &offect_id_shape = (*offset_id_shapes)[i]->GetShapeVector();
      EmbeddingFeatureMappingCheckFeatureAndOffsetIdShape(primitive, feature_id_shape, offect_id_shape, &feature_size);
    }
  } else {
    for (size_t i = start_idx; i < table_num; i++) {
      const auto &feature_id_shape = input_args.at(i)->GetShape()->GetShapeVector();
      const auto &offect_id_shape = input_args.at(i + table_num)->GetShape()->GetShapeVector();
      EmbeddingFeatureMappingCheckFeatureAndOffsetIdShape(primitive, feature_id_shape, offect_id_shape, &feature_size);
    }
  }
  return feature_size;
}
}  // namespace
BaseShapePtr EmbeddingFeatureMappingInsertFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) const {
  return std::make_shared<abstract::TensorShape>(ShapeVector{});
}

TypePtr EmbeddingFeatureMappingInsertFuncImpl::InferType(const PrimitivePtr &primitive,
                                                         const std::vector<AbstractBasePtr> &input_args) const {
  return std::make_shared<TensorType>(kInt32);
}

int32_t EmbeddingFeatureMappingInsertFuncImpl::CheckValidation(const PrimitivePtr &primitive,
                                                               const std::vector<AbstractBasePtr> &input_args) const {
  auto [ret, table_num, feature_size] = CommonCheck(primitive, input_args);
  if (ret != OP_CHECK_SUCCESS) {
    return OP_CHECK_RETRY;
  }
  SetDynInputSizes(primitive, table_num);
  ret = SpecifiedCheck(primitive, input_args, table_num, feature_size);
  return ret;
}

std::tuple<int32_t, size_t, std::vector<int64_t>> EmbeddingFeatureMappingInsertFuncImpl::CommonCheck(
  const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
  const auto &table_name_shape = input_args[table_name_idx_]->GetShape()->GetShapeVector();
  if (MS_UNLIKELY(IsDynamic(table_name_shape))) {
    return std::make_tuple(OP_CHECK_RETRY, kIndex0, std::vector<int64_t>{});
  }

  auto table_num = SizeOf(table_name_shape);
  if (MS_UNLIKELY(input_args.size() != (table_num * kIndex2 + other_arg_num_) &&
                  input_args.size() != (kIndex2 + other_arg_num_))) {
    MS_EXCEPTION(RuntimeError) << "For " << primitive->name() << ", something unexpected happened!";
  }

  int32_t ret = OP_CHECK_SUCCESS;
  auto feature_sie = EmbeddingFeatureMappingCheckFeatureAndOffsetId(primitive, input_args, feature_id_idx_, table_num);
  if (MS_UNLIKELY(feature_sie.size() != table_num)) {
    ret = OP_CHECK_RETRY;
  }

  return std::make_tuple(ret, table_num, std::move(feature_sie));
}

void EmbeddingFeatureMappingInsertFuncImpl::SetDynInputSizes(const PrimitivePtr &primitive, int64_t table_num) const {
  (void)primitive->AddAttr("dyn_input_sizes", MakeValue(std::vector<int64_t>{-1, -1, table_num, table_num}));
}

int32_t EmbeddingFeatureMappingInsertFuncImpl::SpecifiedCheck(const PrimitivePtr &primitive,
                                                              const std::vector<AbstractBasePtr> &input_args,
                                                              size_t table_num,
                                                              const std::vector<int64_t> &feature_size) const {
  return OP_CHECK_SUCCESS;
}
}  // namespace ops
}  // namespace mindspore
