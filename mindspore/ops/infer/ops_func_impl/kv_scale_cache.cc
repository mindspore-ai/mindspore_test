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
#include "infer/ops_func_impl/kv_scale_cache.h"
#include <string>
#include <algorithm>
#include <utility>
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "utils/convert_utils_base.h"
#include "mindspore/ops/ops_utils/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
static constexpr int32_t kPrefillMode = 1;
static constexpr int32_t kIncrementalMode = 0;
}  // namespace
BaseShapePtr KvScaleCacheFuncImpl::InferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) const {
  auto op_name = primitive->name();

  auto key_scale_shape_ptr = input_args[KvScaleCacheInputKeyScaleIndex]->GetShape();
  auto value_scale_shape_ptr = input_args[KvScaleCacheInputValueScaleIndex]->GetShape();
  MS_EXCEPTION_IF_NULL(key_scale_shape_ptr);
  MS_EXCEPTION_IF_NULL(value_scale_shape_ptr);

  auto key_scale_cache_shape_ptr = input_args[KvScaleCacheInputKeyValueScaleCacheIndex]->GetShape();
  MS_EXCEPTION_IF_NULL(key_scale_cache_shape_ptr);
  auto key_scale_cache_shape = key_scale_cache_shape_ptr->GetShapeVector();

  auto batch_valid_shape_ptr = input_args[KvScaleCacheInputBatchVaildLengthIndex]->GetShape();
  MS_EXCEPTION_IF_NULL(batch_valid_shape_ptr);
  auto batch_valid_shape = batch_valid_shape_ptr->GetShapeVector();

  if (IsDynamicRank(key_scale_shape_ptr->GetShapeVector()) || IsDynamicRank(batch_valid_shape_ptr->GetShapeVector())) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }

  const int64_t input_num_dims = 2;
  MS_CHECK_VALUE(
    key_scale_shape_ptr->GetShapeVector().size() == input_num_dims,
    CheckAndConvertUtils::FormatCheckIntegerMsg(
      "rank of kscale", SizeToLong(key_scale_shape_ptr->GetShapeVector().size()), kEqual, input_num_dims, primitive));
  MS_CHECK_VALUE(
    value_scale_shape_ptr->GetShapeVector().size() == input_num_dims,
    CheckAndConvertUtils::FormatCheckIntegerMsg(
      "rank of vscale", SizeToLong(value_scale_shape_ptr->GetShapeVector().size()), kEqual, input_num_dims, primitive));

  const size_t batch_valid_size = batch_valid_shape.size();
  (void)CheckAndConvertUtils::CheckInteger("batch_valid_size must be greater than 0, but got:", batch_valid_size,
                                           kGreaterEqual, 0, op_name);

  if (!IsDynamic(key_scale_cache_shape) && !IsDynamic(batch_valid_shape)) {
    const size_t key_scale_cache_dim = key_scale_cache_shape[0];
    const size_t max_batch_size = key_scale_cache_shape[1];
    // max_batch_size 约束
    MS_CHECK_VALUE(batch_valid_size <= max_batch_size,
                   CheckAndConvertUtils::FormatCommMsg(
                     "The batch_size must not bigger than max_batch_size, but got batch_valid_size: ", batch_valid_size,
                     ", max_batch_size: ", max_batch_size));
    MS_CHECK_VALUE(key_scale_cache_dim == input_num_dims,
                   CheckAndConvertUtils::FormatCheckIntegerMsg("key_scale_cache_dim", SizeToLong(key_scale_cache_dim),
                                                               kEqual, input_num_dims, primitive));
    MS_CHECK_VALUE(batch_valid_size != 0, CheckAndConvertUtils::FormatCheckIntegerMsg(
                                            "batch_valid_size", SizeToLong(batch_valid_size), kNotEqual, 0, primitive));
    // max_seqlens约束
    const size_t max_seqlens = key_scale_cache_shape[2];
    MS_CHECK_VALUE(max_seqlens != 0, CheckAndConvertUtils::FormatCheckIntegerMsg("max_seqlens", SizeToLong(max_seqlens),
                                                                                 kNotEqual, 0, primitive));
    auto batch_valid_tensor = input_args[KvScaleCacheInputBatchVaildLengthIndex];
    // 获取 batch_valid_length 的最大值
    if (batch_valid_tensor->GetValue() != nullptr) {
      auto shape_ptr = batch_valid_tensor->GetShape()->cast<abstract::ShapePtr>();
      MS_EXCEPTION_IF_NULL(shape_ptr);
      const auto &shape = shape_ptr->shape();
      auto max_value = *std::max_element(shape.begin(), shape.end());
      MS_CHECK_VALUE(max_value <= static_cast<int64_t>(max_seqlens),
                     CheckAndConvertUtils::FormatCommMsg("Max seqlen in batch exceeds limit:", max_value,
                                                         " > max_seqlens:", max_seqlens));
    }
  }

  // decode-check
  auto cache_mode_scalar = GetScalarValue<int64_t>(input_args[KvScaleCacheInputCacheModeIndex]->GetValue());
  size_t decode_batch = key_scale_shape_ptr->GetShapeVector()[0];
  size_t seqlens = key_scale_shape_ptr->GetShapeVector()[1];
  if (cache_mode_scalar.has_value()) {
    auto cache_mode = static_cast<int32_t>(cache_mode_scalar.value());
    MS_LOG(DEBUG) << "cache_mode: " << cache_mode;
    if (cache_mode != kIncrementalMode && cache_mode != kPrefillMode && cache_mode != -1) {
      MS_LOG(EXCEPTION) << "this cache_mode is not supported, but got cache_mode: " << cache_mode;
    }
    if (cache_mode == kIncrementalMode) {
      MS_CHECK_VALUE(
        (decode_batch >= batch_valid_size) && (seqlens == 1),
        CheckAndConvertUtils::FormatCommMsg(
          "For '", op_name, "', decode_batch must be more than or equal to batch_valid_size, seqlens must be 1"));
    }
  }

  auto shape_element = key_scale_cache_shape_ptr->cast<abstract::ShapePtr>();
  return shape_element;
}

TypePtr KvScaleCacheFuncImpl::InferType(const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) const {
  const std::set<TypePtr> valid_types = {kFloat32};
  auto op_name = primitive->name();
  std::map<std::string, TypePtr> types;

  (void)types.emplace("key_scale", input_args[KvScaleCacheInputKeyScaleIndex]->GetType());
  (void)types.emplace("value_scale", input_args[KvScaleCacheInputValueScaleIndex]->GetType());
  (void)types.emplace("key_value_scale_cache", input_args[KvScaleCacheInputKeyValueScaleCacheIndex]->GetType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);

  auto bvl_type = input_args[KvScaleCacheInputBatchVaildLengthIndex]->GetType();
  const std::set<TypePtr> int32_valid_types = {kInt32};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("batch_valid_length", bvl_type, int32_valid_types, op_name);
  return type;
}
}  // namespace ops
}  // namespace mindspore
