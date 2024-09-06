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

#include "infer/ops_func_impl/paged_attention.h"
#include "utils/check_convert_utils.h"
#include "infer/ops_func_impl/common_infer_fns.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
BaseShapePtr PagedAttentionFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) const {
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kPagedAttentionInputsNum, primitive->name());

  auto query_shape_ptr = input_args[kPagedAttentionInputQueryIndex]->GetShape();
  auto shape_element = query_shape_ptr->cast<abstract::ShapePtr>();
  return shape_element;
}

TypePtr PagedAttentionFuncImpl::InferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  bool enable_infer_boost = ms_context->IsEnableInferBoost();
  auto op_name = primitive->name();

  std::set<TypePtr> valid_types = {kFloat16};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("query", input_args[kPagedAttentionInputQueryIndex]->GetType());

  auto key_type = input_args[kPagedAttentionInputKeyCacheIndex]->GetType();
  auto value_type = input_args[kPagedAttentionInputValueCacheIndex]->GetType();

  auto key_tensor_type = key_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(key_tensor_type);
  bool kvcache_quant = (key_tensor_type->element()->type_id() == TypeId::kNumberTypeInt8);
  if (kvcache_quant && enable_infer_boost) {
    //  infer_boost support int8 kv_cache when query dtype is fp16
    std::map<std::string, TypePtr> kvcache_types;
    (void)kvcache_types.emplace("key_cache", key_type);
    (void)kvcache_types.emplace("value_cache", value_type);
    (void)CheckAndConvertUtils::CheckTensorTypeSame(kvcache_types, {kInt8}, op_name);
  } else {
    // else q, k, v should have same dtypes, fp16 or bf16
    (void)valid_types.emplace(kBFloat16);
    (void)types.emplace("key_cache", key_type);
    (void)types.emplace("value_cache", value_type);
  }
  auto output_dtype = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);

  //  check antiquant scale and offset's dtype when they are NOT None
  if (enable_infer_boost && !IsOptionalInputNone(input_args[kPagedAttentionInputAntiquantScaleIndex]) &&
      !IsOptionalInputNone(input_args[kPagedAttentionInputAntiquantOffsetIndex])) {
    bool valid_flag = false;
    auto scale_type = input_args[kPagedAttentionInputAntiquantScaleIndex]->GetType()->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(scale_type);
    auto scale_type_id = scale_type->element()->type_id();

    auto offset_type = input_args[kPagedAttentionInputAntiquantOffsetIndex]->GetType()->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(offset_type);
    auto offset_type_id = offset_type->element()->type_id();

    if ((scale_type_id == TypeId::kNumberTypeFloat16 && offset_type_id == TypeId::kNumberTypeFloat16) ||
        (scale_type_id == TypeId::kNumberTypeInt64 && offset_type_id == TypeId::kNumberTypeInt32)) {
      valid_flag = true;
    }
    if (!valid_flag) {
      MS_LOG(EXCEPTION) << "types of antiquant_scale & antiquant_offset are not supported: "
                        << input_args[kPagedAttentionInputAntiquantScaleIndex]->GetType() << " & "
                        << input_args[kPagedAttentionInputAntiquantOffsetIndex]->GetType();
    }
  }

  auto block_tables_type = input_args[kPagedAttentionInputBlockTablesIndex]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("block_tables", block_tables_type, {kInt32, kInt64, kUInt64},
                                                   op_name);
  auto context_lens_type = input_args[kPagedAttentionInputContextLensIndex]->GetType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("context_lens", context_lens_type, {kInt32, kInt64, kUInt64},
                                                   op_name);

  return output_dtype;  // attention_out dtype
}
}  // namespace ops
}  // namespace mindspore
