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

#include <map>
#include <string>
#include <set>
#include <utility>

#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/helper.h"
#include "op_def/comparison_ops.h"
#include "infer/ops_func_impl/incre_flash_attention.h"
#include "infer/ops_func_impl/common_infer_fns.h"
#include "ops_utils/op_utils.h"
#include "op_def/op_enum.h"
#include "utils/check_convert_utils.h"
#include "ops/ops_func_impl/simple_infer.h"

namespace mindspore {
namespace ops {
constexpr size_t kInputQueryBSHRank = 3;
constexpr size_t kInputQueryBNSDRank = 4;

void CheckKeyValueList(const AbstractBasePtr &arg, const string &op_name, const string &arg_name) {
  size_t valid_seq_length = 1;
  if (!(arg->isa<abstract::AbstractSequence>() &&
        arg->cast<abstract::AbstractSequencePtr>()->elements().size() == valid_seq_length &&
        arg->cast<abstract::AbstractSequencePtr>()->elements()[kIndex0]->isa<abstract::AbstractTensor>())) {
    MS_LOG(EXCEPTION) << op_name << ": parameter " << arg_name << " should be a sequence containing exactly 1 tensor. ";
  }
}

ShapeValueDType GetDimension(const std::vector<ShapeValueDType> &dimensions, const std::string &op_name,
                             const std::string &input_name) {
  if (dimensions.empty()) {
    return abstract::Shape::kShapeDimAny;
  }
  ShapeValueDType baseValue = abstract::Shape::kShapeDimAny;
  for (const auto &item : dimensions) {
    if (item == abstract::Shape::kShapeDimAny || item == baseValue) {
      continue;
    }
    if (baseValue == abstract::Shape::kShapeDimAny && item > 0) {
      baseValue = item;
    } else {
      std::ostringstream buffer;
      for (const auto &dim : dimensions) {
        buffer << dim << ", ";
      }
      MS_LOG(EXCEPTION) << "For primitive[" << op_name << "], the " << input_name << " should not be equal -1 or equal "
                        << baseValue << " but got " << buffer.str();
    }
  }
  return baseValue;
}

void CheckInputsShape(const AbstractBasePtr &input, const std::vector<ShapeValueDType> &expect_shape,
                      const std::string &op_name, const std::string &input_name, bool optional = false) {
  MS_EXCEPTION_IF_NULL(input);
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input->GetShape())[kShape];
  if (optional && input_shape.empty()) {
    return;
  }
  if (input_shape != expect_shape) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input `" << input_name << "' must be -- " << expect_shape
                      << ", but got shape is " << input_shape;
  }
}

void ParamsValidCheck(const PrimitivePtr &primitive, const std::vector<int64_t> &query_shape,
                      const std::vector<int64_t> &key_shape, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  auto Q_H = query_shape[2];
  auto KV_H = key_shape[2];
  auto num_heads_arg = input_args[kIncreFlashAttentionInputNumHeads];
  auto num_key_value_heads_arg = input_args[kIncreFlashAttentionInputNumKeyValueHeads];
  auto num_heads_opt = GetScalarValue<int64_t>(num_heads_arg->GetValue());
  auto num_key_value_heads_opt = GetScalarValue<int64_t>(num_key_value_heads_arg->GetValue());
  if (!num_heads_opt.has_value() || !num_key_value_heads_opt.has_value()) {
    return;
  }
  auto N = num_heads_opt.value();
  auto KV_N = num_key_value_heads_opt.value();
  if (Q_H % N != 0) {
    MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << Q_H << " and " << N;
  }
  if (KV_N != 0) {
    if (KV_H % KV_N != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'kv_hidden_size` must be divisible by `num_key_value_heads`, but got " << KV_H
                        << " and " << KV_N;
    }
    if (N % KV_N != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'num_heads` must be divisible by `num_key_value_heads`, but got " << N
                        << " and " << KV_N;
    }
  } else {
    if (KV_H % N != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'kv_hidden_size` must be divisible by `head_num`, but got " << KV_H << " and "
                        << N;
    }
  }
}

void CheckShapeSizeRight(const PrimitivePtr &primitive, size_t shape_size) {
  auto op_name = primitive->name();
  if (shape_size != kInputQueryBSHRank && shape_size != kInputQueryBNSDRank) {
    MS_LOG(EXCEPTION) << op_name << ": key or value's shape must be 3 or 4, but got " << shape_size;
  }
}

bool CheckIsFrontend(const std::vector<AbstractBasePtr> &input_args) {
  if (input_args[kIncreFlashAttentionInputKeyIndex]->isa<abstract::AbstractSequence>()) {
    return true;
  }
  return false;
}

std::vector<int64_t> ObtainCorrShape(const std::vector<AbstractBasePtr> &input_args, size_t index) {
  std::vector<int64_t> out_shape;
  if (CheckIsFrontend(input_args)) {
    out_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[index]->GetShape())[kShape];
  } else {
    AbstractBasePtrList elements = input_args;
    out_shape = elements[index]->GetShape()->GetShapeVector();
  }
  return out_shape;
}

std::vector<int64_t> GetIFADynInputShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                         size_t index) {
  const auto &prim_name = primitive->name();
  std::vector<int64_t> query_shape = ObtainCorrShape(input_args, kIncreFlashAttentionInputQueryIndex);
  auto input_layout_arg = input_args[kIncreFlashAttentionInputInputLayout];
  auto input_layout_opt = GetScalarValue<int64_t>(input_layout_arg->GetValue());
  if (!input_layout_opt.has_value()) {
    std::vector<int64_t> dyn_output{abstract::Shape::kShapeRankAny};
    return dyn_output;
  }
  FASInputLayoutMode input_layout_value = static_cast<FASInputLayoutMode>(input_layout_opt.value());
  if (!CheckIsFrontend(input_args)) {
    std::vector<int64_t> shape_vec = ObtainCorrShape(input_args, index);
    if (IsDynamicRank(shape_vec)) {
      if (input_layout_value == BSH) {
        return std::vector(kInputQueryBSHRank, abstract::Shape::kShapeDimAny);
      } else {
        return std::vector(kInputQueryBNSDRank, abstract::Shape::kShapeDimAny);
      }
    }
    return shape_vec;
  }

  AbstractBasePtrList kv_elements = input_args[index]->cast<abstract::AbstractSequencePtr>()->elements();

  // if dyn rank
  auto ele_first = kv_elements[kIndex0]->cast<abstract::AbstractTensorPtr>();
  std::vector<int64_t> ele_first_sp = CheckAndConvertUtils::ConvertShapePtrToShapeMap(ele_first->BuildShape())[kShape];
  if (IsDynamicRank(ele_first_sp)) {
    if (input_layout_value == BSH) {
      return std::vector(kInputQueryBSHRank, abstract::Shape::kShapeDimAny);
    } else {
      return std::vector(kInputQueryBNSDRank, abstract::Shape::kShapeDimAny);
    }
  }

  if (kv_elements.size() == 1) {  // [B S H]
    auto element0 = kv_elements[kIndex0]->cast<abstract::AbstractTensorPtr>();
    std::vector<int64_t> element0_sp = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element0->BuildShape())[kShape];
    CheckShapeSizeRight(primitive, element0_sp.size());
    return element0_sp;
  }
  if (!IsDynamicRank(query_shape) && !IsDynamicShape(query_shape) && (int64_t)kv_elements.size() != query_shape[0]) {
    MS_EXCEPTION(TypeError) << "For '" << prim_name
                            << "', the key or value's list length must be B. But got:" << kv_elements.size();
  }
  std::vector<int64_t> element_first_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    kv_elements[kIndex0]->cast<abstract::AbstractTensorPtr>()->BuildShape())[kShape];
  CheckShapeSizeRight(primitive, element_first_shape.size());

  std::vector<int64_t> element_each_shape;
  for (size_t i = 0; i < kv_elements.size(); ++i) {
    auto element_each = kv_elements[i]->cast<abstract::AbstractTensorPtr>();
    element_each_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(element_each->GetShape())[kShape];
    if (element_each_shape != element_first_shape) {
      MS_LOG(EXCEPTION) << prim_name << ": each element of key or value should be the same shape";
    }
  }
  element_first_shape[kIndex0] = (int64_t)kv_elements.size();
  return element_first_shape;
}

void CheckPaddingAttenMaskShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                int64_t B, int64_t S) {
  auto op_name = primitive->name();
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputPseShiftIndex])) {
    std::vector<int64_t> pse_shift_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
      input_args[kIncreFlashAttentionInputPseShiftIndex]->GetShape())[kShape];
    size_t len_pa = pse_shift_shape.size();
    if (len_pa > 0 && !pse_shift_shape.empty() && !IsDynamicShape(pse_shift_shape)) {
      if ((pse_shift_shape[0] != B && pse_shift_shape[0] != 1) || pse_shift_shape[len_pa - 1] != S) {
        MS_LOG(EXCEPTION) << op_name << ": The shape of pse_shift must be: "
                          << "(B ... S) or (1 ... S)"
                          << ", but got shape (" << pse_shift_shape[0] << " ... " << pse_shift_shape[len_pa - 1] << ")";
      }
    }
  }

  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputAttnMaskIndex]) &&
      IsOptionalInputNone(input_args[kIncreFlashAttentionInputBlockTable])) {
    std::vector<int64_t> atten_mask_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
      input_args[kIncreFlashAttentionInputAttnMaskIndex]->GetShape())[kShape];
    size_t len_pa = atten_mask_shape.size();
    if (len_pa > 0 && !atten_mask_shape.empty() && !IsDynamicShape(atten_mask_shape)) {
      if (atten_mask_shape[0] != B || atten_mask_shape[len_pa - 1] != S) {
        MS_LOG(EXCEPTION) << op_name << ": The shape of atten_mask must be: "
                          << "(B ... S)"
                          << ", but got shape is (" << atten_mask_shape[0] << " ... " << atten_mask_shape[len_pa - 1]
                          << ")";
      }
    }
  }
}

void CheckActualSeqLengthsShapeValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args,
                                     int64_t B, int64_t S) {
  if (IsOptionalInputNone(input_args[kIncreFlashAttentionInputActualSeqLengths])) {
    return;
  }
  auto op_name = primitive->name();
  auto asl_type = input_args[kIncreFlashAttentionInputActualSeqLengths]->GetType();
  MS_EXCEPTION_IF_NULL(asl_type);
  if (!asl_type->isa<TensorType>()) {
    return;
  }
  std::vector<int64_t> asl_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(
    input_args[kIncreFlashAttentionInputActualSeqLengths]->GetShape())[kShape];
  if (IsDynamic(asl_shape)) {
    return;
  }
  if (asl_shape.size() != 1 || (asl_shape[0] != 1 && asl_shape[0] != B)) {
    MS_LOG(EXCEPTION) << op_name << ": The size of actual_seq_lengths's shape must be: 1 or " << B << ", but got "
                      << asl_shape[0];
  }
}

abstract::ShapePtr IncreFlashAttentionInferShapeBSH(const PrimitivePtr &primitive,
                                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kLessEqual, kIncreFlashAttentionInputsNum, op_name);
  std::vector<int64_t> query_shape = ObtainCorrShape(input_args, kIncreFlashAttentionInputQueryIndex);
  std::vector<int64_t> key_shape = GetIFADynInputShape(primitive, input_args, kIncreFlashAttentionInputKeyIndex);
  std::vector<int64_t> value_shape = GetIFADynInputShape(primitive, input_args, kIncreFlashAttentionInputValueIndex);

  if (IsDynamicRank(query_shape)) {
    query_shape = std::vector(kInputQueryBSHRank, abstract::Shape::kShapeDimAny);
  }

  if (CheckIsFrontend(input_args) && IsOptionalInputNone(input_args[kIncreFlashAttentionInputBlockTable])) {
    if (!IsDynamicShape(query_shape) && !IsDynamicShape(key_shape) && !IsDynamicShape(value_shape)) {
      int64_t B = query_shape[0];
      int64_t Q_H = query_shape[2];
      ParamsValidCheck(primitive, query_shape, key_shape, input_args);
      CheckInputsShape(input_args[kIncreFlashAttentionInputQueryIndex], {B, 1, Q_H}, op_name, "query");
      int64_t S = key_shape[1];
      CheckPaddingAttenMaskShape(primitive, input_args, B, S);
      CheckActualSeqLengthsShapeValue(primitive, input_args, B, S);
      if (key_shape != value_shape) {
        MS_LOG(EXCEPTION) << op_name << ": The shape of key and value must be same, but got: " << key_shape << " and "
                          << value_shape;
      }
    }
  }

  ShapeVector attention_out_shape(kInputQueryBSHRank, abstract::Shape::kShapeDimAny);
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputBlockTable])) {
    // kv: [num_blocks,block_size,hidden_size], q: [batch,seq_length,hidden_size]
    attention_out_shape[0] = query_shape[0];
  } else {
    attention_out_shape[0] = GetDimension({query_shape[0]}, op_name, "B");
  }
  attention_out_shape[1] = 1;
  attention_out_shape[2] = GetDimension({query_shape[2]}, op_name, "H");  // 2: h_index
  return std::make_shared<abstract::Shape>(attention_out_shape);
}

abstract::ShapePtr IncreFlashAttentionInferShapeBNSD(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kLessEqual, kIncreFlashAttentionInputsNum, op_name);
  std::vector<int64_t> query_shape = ObtainCorrShape(input_args, kIncreFlashAttentionInputQueryIndex);
  std::vector<int64_t> key_shape = GetIFADynInputShape(primitive, input_args, kIncreFlashAttentionInputKeyIndex);
  std::vector<int64_t> value_shape = GetIFADynInputShape(primitive, input_args, kIncreFlashAttentionInputValueIndex);
  if (IsDynamicRank(query_shape)) {
    query_shape = std::vector(kInputQueryBNSDRank, abstract::Shape::kShapeDimAny);
  }
  if (CheckIsFrontend(input_args) && IsOptionalInputNone(input_args[kIncreFlashAttentionInputBlockTable])) {
    if (!IsDynamicShape(query_shape) && !IsDynamicShape(key_shape) && !IsDynamicShape(value_shape)) {
      int64_t B = query_shape[0];
      int64_t N_Q = query_shape[1];
      int64_t D = query_shape[3];
      int64_t KV_N = key_shape[1];
      auto num_heads_arg = input_args[kIncreFlashAttentionInputNumHeads];
      auto num_key_value_heads_arg = input_args[kIncreFlashAttentionInputNumKeyValueHeads];
      auto num_heads_opt = GetScalarValue<int64_t>(num_heads_arg->GetValue());
      auto num_key_value_heads_opt = GetScalarValue<int64_t>(num_key_value_heads_arg->GetValue());
      if (!num_heads_opt.has_value() || !num_key_value_heads_opt.has_value()) {
        ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
        return std::make_shared<abstract::Shape>(std::move(dyn_output));
      }
      int64_t N = num_heads_opt.value();
      int64_t KV_N_ATTR = num_key_value_heads_opt.value();
      if (N_Q != N) {
        MS_LOG(EXCEPTION) << op_name << ": query 's shape[1] should be num_heads, but got: " << N_Q << " and " << N;
      }
      if (KV_N_ATTR != 0 && KV_N != KV_N_ATTR) {
        MS_LOG(EXCEPTION) << op_name << ": key and value 's shape[1] should be num_key_value_heads, but got: " << KV_N
                          << " and " << KV_N_ATTR;
      }
      if (N % KV_N != 0) {
        MS_LOG(EXCEPTION) << op_name << ": 'num_heads` must be divisible by `num_key_value_heads`, but got " << N
                          << " and " << KV_N;
      }
      CheckInputsShape(input_args[kIncreFlashAttentionInputQueryIndex], {B, N, 1, D}, op_name, "query");

      int64_t S = key_shape[2];
      CheckPaddingAttenMaskShape(primitive, input_args, B, S);
      CheckActualSeqLengthsShapeValue(primitive, input_args, B, S);
      if (key_shape != value_shape) {
        MS_LOG(EXCEPTION) << op_name << ": The shape of key and value must be same, but got: " << key_shape << " and "
                          << value_shape;
      }
    }
  }

  ShapeVector attention_out_shape(kInputQueryBNSDRank, abstract::Shape::kShapeDimAny);
  attention_out_shape[0] = GetDimension({query_shape[0]}, op_name, "B");
  attention_out_shape[1] = GetDimension({query_shape[1]}, op_name, "N");
  attention_out_shape[2] = 1;                                             // 2: s_index
  attention_out_shape[3] = GetDimension({query_shape[3]}, op_name, "D");  // 3: d_index
  return std::make_shared<abstract::Shape>(attention_out_shape);
}

BaseShapePtr IncreFlashAttentionInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  auto input_layout_arg = input_args[kIncreFlashAttentionInputInputLayout];
  auto input_layout_opt = GetScalarValue<int64_t>(input_layout_arg->GetValue());
  if (!input_layout_opt.has_value()) {
    ShapeVector dyn_output{abstract::Shape::kShapeRankAny};
    return std::make_shared<abstract::Shape>(std::move(dyn_output));
  }
  FASInputLayoutMode input_layout_value = static_cast<FASInputLayoutMode>(input_layout_opt.value());
  if (input_layout_value == BSH) {
    return IncreFlashAttentionInferShapeBSH(primitive, input_args);
  } else {
    return IncreFlashAttentionInferShapeBNSD(primitive, input_args);
  }
}

BaseShapePtr IncreFlashAttentionFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) const {
  if (CheckIsFrontend(input_args)) {
    auto op_name = primitive->name();
    CheckKeyValueList(input_args[kIncreFlashAttentionInputKeyIndex], op_name, "key");
    CheckKeyValueList(input_args[kIncreFlashAttentionInputValueIndex], op_name, "value");
  }
  return IncreFlashAttentionInferShape(primitive, input_args);
}

void CheckQuantParamType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = prim->name();
  // std::map<std::string, TypePtr> dequant_types;
  // const std::set<TypePtr> dequant_valid_types = {kUInt64};
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputDequantScale1])) {
    MS_LOG(EXCEPTION) << "dequant_scale1 is not support now. It must be None.";
  }
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputDequantScale2])) {
    MS_LOG(EXCEPTION) << "dequant_scale2 is not support now. It must be None.";
  }
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputQuantScale1])) {
    MS_LOG(EXCEPTION) << "quant_scale1 is not support now. It must be None.";
  }
  const std::set<TypePtr> quant_valid_types = {kFloat32};
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputQuantScale2])) {
    auto quant_scale2_type = input_args[kIncreFlashAttentionInputQuantScale2]->GetType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("quant_scale2", quant_scale2_type, quant_valid_types, op_name);
  }
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputQuantOffset2])) {
    auto quant_offset2_type = input_args[kIncreFlashAttentionInputQuantOffset2]->GetType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("quant_offset2", quant_offset2_type, quant_valid_types, op_name);
  }
  const std::set<TypePtr> antiquant_valid_types = {kFloat16, kBFloat16};
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputAntiquantScale])) {
    auto antiquant_scale_type = input_args[kIncreFlashAttentionInputAntiquantScale]->GetType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("antiquant_scale", antiquant_scale_type, antiquant_valid_types,
                                                     op_name);
  }
  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputAntiquantOffset])) {
    auto antiquant_offset_type = input_args[kIncreFlashAttentionInputAntiquantOffset]->GetType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("antiquant_offset", antiquant_offset_type, antiquant_valid_types,
                                                     op_name);
  }
}

TypePtr IncreFlashAttentionInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  if (CheckIsFrontend(input_args)) {
    CheckQuantParamType(primitive, input_args);
    const std::set<TypePtr> pse_shift_valid_types = {kFloat16, kBFloat16};
    if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputPseShiftIndex])) {
      auto pse_shift_type = input_args[kIncreFlashAttentionInputPseShiftIndex]->GetType();
      (void)CheckAndConvertUtils::CheckTensorTypeValid("pse_shift", pse_shift_type, pse_shift_valid_types, op_name);
    }
    const std::set<TypePtr> atten_mask_valid_types = {kBool, kInt8, kUInt8};
    if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputAttnMaskIndex])) {
      auto attn_mask_type = input_args[kIncreFlashAttentionInputAttnMaskIndex]->GetType();
      (void)CheckAndConvertUtils::CheckTensorTypeValid("attn_mask", attn_mask_type, atten_mask_valid_types, op_name);
    }
    auto asl_type = input_args[kIncreFlashAttentionInputActualSeqLengths]->GetType();
    MS_EXCEPTION_IF_NULL(asl_type);
    if (asl_type->isa<TensorType>()) {
      const std::set<TypePtr> acl_valid_types = {kInt64, kInt32};
      if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputActualSeqLengths])) {
        auto actual_seq_lengths_type = input_args[kIncreFlashAttentionInputActualSeqLengths]->GetType();
        (void)CheckAndConvertUtils::CheckTensorTypeValid("actual_seq_lengths", actual_seq_lengths_type, acl_valid_types,
                                                         op_name);
      }
    }
    const std::set<TypePtr> block_table_valid_types = {kInt32};
    if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputBlockTable])) {
      auto block_table_type = input_args[kIncreFlashAttentionInputBlockTable]->GetType();
      (void)CheckAndConvertUtils::CheckTensorTypeValid("block_table", block_table_type, block_table_valid_types,
                                                       op_name);
    }
    const std::set<TypePtr> kv_padding_size_valid_types = {kInt64};
    if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputKvPaddingSize])) {
      auto kv_padding_size_type = input_args[kIncreFlashAttentionInputKvPaddingSize]->GetType();
      (void)CheckAndConvertUtils::CheckTensorTypeValid("kv_padding_size", kv_padding_size_type,
                                                       kv_padding_size_valid_types, op_name);
    }
  }

  if (!IsOptionalInputNone(input_args[kIncreFlashAttentionInputQuantScale2])) {
    return kInt8;
  }
  std::map<std::string, TypePtr> kv_types;
  const std::set<TypePtr> kv_valid_types = {kFloat16, kBFloat16, kInt8};
  TypePtr type;
  if (CheckIsFrontend(input_args)) {
    AbstractBasePtrList elements =
      input_args[kIncreFlashAttentionInputKeyIndex]->cast<abstract::AbstractSequencePtr>()->elements();
    (void)kv_types.emplace("key", elements[kIndex0]->GetType());
    elements = input_args[kIncreFlashAttentionInputValueIndex]->cast<abstract::AbstractSequencePtr>()->elements();
    (void)kv_types.emplace("value", elements[kIndex0]->GetType());
    type = CheckAndConvertUtils::CheckTensorTypeSame(kv_types, kv_valid_types, op_name);
    MS_EXCEPTION_IF_NULL(type);
  }
  const std::set<TypePtr> q_valid_types = {kFloat16, kBFloat16};
  auto query_type = input_args[kIncreFlashAttentionInputQueryIndex]->GetType();
  type = CheckAndConvertUtils::CheckTensorTypeValid("query", query_type, q_valid_types, op_name);
  return type;
}

TypePtr IncreFlashAttentionFuncImpl::InferType(const PrimitivePtr &primitive,
                                               const std::vector<AbstractBasePtr> &input_args) const {
  if (CheckIsFrontend(input_args)) {
    auto op_name = primitive->name();
    CheckKeyValueList(input_args[kIncreFlashAttentionInputKeyIndex], op_name, "key");
    CheckKeyValueList(input_args[kIncreFlashAttentionInputValueIndex], op_name, "value");
  }
  return IncreFlashAttentionInferType(primitive, input_args);
}

std::set<int64_t> IncreFlashAttentionFuncImpl::GetValueDependArgIndices() const {
  return {kIncreFlashAttentionInputActualSeqLengths};
}  // 4: pos of valuedepend

TypePtrList IncreFlashAttentionFuncImpl::InferType(const PrimitivePtr &primitive,
                                                   const ValuePtrList &input_values) const {
  if (input_values[kIncreFlashAttentionInputQuantScale2] != mindspore::kNone) {
    return {kInt8};
  }
  if (input_values[kIncreFlashAttentionInputKvPaddingSize]->isa<tensor::BaseTensor>()) {
    const auto &kv_padding_size_tensor =
      input_values[kIncreFlashAttentionInputKvPaddingSize]->cast<tensor::BaseTensorPtr>();
    const std::set<TypePtr> kv_padding_size_valid_types = {kInt64};
    (void)CheckAndConvertUtils::CheckTypeValid("kv_padding_size", kv_padding_size_tensor->Dtype(),
                                               kv_padding_size_valid_types, primitive->name());
  }
  const auto &query_tensor = input_values[kIncreFlashAttentionInputQueryIndex]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(query_tensor);
  return {query_tensor->Dtype()};
}

ShapeArray IncreFlashAttentionFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const ValuePtrList &input_values) const {
  const auto &query_tensor = input_values[kIncreFlashAttentionInputQueryIndex]->cast<tensor::BaseTensorPtr>();
  MS_EXCEPTION_IF_NULL(query_tensor);
  return {query_tensor->shape()};
}
REGISTER_SIMPLE_INFER(kNameIncreFlashAttention, IncreFlashAttentionFuncImpl)

}  // namespace ops
}  // namespace mindspore
