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

#include "infer/ops_func_impl/flash_attention_score.h"

#include <string>
#include <map>
#include <memory>
#include <array>
#include <algorithm>

#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "mindapi/helper.h"
#include "ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
constexpr size_t kFlashAttentionScoreSoftmaxLastDim = 8;
constexpr size_t kInputFlashAttentionScoreQueryBSHRank = 3;
constexpr size_t kInputFlashAttentionScoreQuerySBHRank = 3;
constexpr size_t kInputFlashAttentionScoreQueryTNDRank = 3;
constexpr size_t kInputFlashAttentionScoreQueryBNSDRank = 4;
constexpr size_t kInputFlashAttentionScoreQueryBSNDRank = 4;
constexpr size_t kFAGRealShiftCompressionDim = 1024;
constexpr size_t kInputFlashAttentionScoreAttnMaskCompressionDim = 2048;
constexpr auto kEnableRingAttention = "enable_ring_attention";
constexpr auto kEnableFlashSP = "enable_flash_sp";
constexpr auto kEnableRASendRecv = "enable_ra_send_recv";

namespace {
static const std::array<int64_t, 7> kNeedCompressAttnMaskMode = {
  kSparseLeftUpCausal, kSparseRightDownCausal, kSparseBand,      kSparsePrefix,
  kSparseGlobal,       kSparseDilated,         kSparseBlockLocal};
}

// None indicates that the optional input is not passed
bool IsFlashAttentionScoreOptionalInputNotPass(const InferInfoPtr &input) { return input->IsNone(); }

static inline void EnsurePaddingMaskNone(const InferInfoPtr &padding_mask, const std::string &op_name) {
  if (!IsFlashAttentionScoreOptionalInputNotPass(padding_mask)) {
    MS_LOG(EXCEPTION) << op_name << ": 'padding_mask' must be None currently.";
  }
}

static inline void ValidateKeepProbAndDropMask(const InferInfoPtrList &input_infos, const std::string &op_name) {
  auto keep_prob_opt = input_infos[kFlashAttentionScoreInputKeepProbIndex]->GetScalarValue<float>();
  if (!keep_prob_opt.has_value()) {
    return;
  }
  const auto keep_prob = keep_prob_opt.value();
  if (keep_prob > 1 || keep_prob < 0) {
    MS_LOG(EXCEPTION) << op_name << ": attribute `keep_prob` must be a floating point number in [0, 1], but got "
                      << keep_prob;
  }
  if (common::IsFloatEqual(keep_prob, 1.0)) {
    if (!IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputDropMaskIndex])) {
      MS_LOG(EXCEPTION) << op_name << ": 'drop_mask' must be None when keep_prob is 1.0.";
    }
    return;
  }
  if (!IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputDropMaskIndex])) {
    const auto drop_mask_type = input_infos[kFlashAttentionScoreInputDropMaskIndex]->GetType();
    CheckAndConvertUtils::CheckTypeIdValid("drop_mask", drop_mask_type, {kNumberTypeUInt8}, op_name);
  }
}

void CheckFlashAttentionScoreInputShape(const InferInfoPtr &input, const ShapeVector &expect_shape,
                                        const std::string &op_name, const std::string &input_name,
                                        bool optional = false) {
  if (IsFlashAttentionScoreOptionalInputNotPass(input) && optional) {
    return;
  }
  const auto input_shape = input->GetShape();
  if (input_shape != expect_shape) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input `" << input_name << "' must be " << expect_shape
                      << ", but got shape is " << input_shape;
  }
}

void CheckFlashAttentionScoreInputShape(const InferInfoPtr &input, const std::vector<ShapeVector> &expect_shape_list,
                                        const std::string &op_name, const std::string &input_name,
                                        bool optional = false) {
  if (IsFlashAttentionScoreOptionalInputNotPass(input) && optional) {
    return;
  }
  const auto input_shape = input->GetShape();
  if (std::all_of(expect_shape_list.begin(), expect_shape_list.end(),
                  [&input_shape](const ShapeVector &expect_shape) { return input_shape != expect_shape; })) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input " << input_name << " must be one of " << expect_shape_list
                      << ", but got shape is " << input_shape;
  }
}

void CheckFlashAttentionScoreAttnMaskShape(const InferInfoPtr &attn_mask, const std::string &op_name,
                                           int64_t sparse_mode, int64_t batch_size, int64_t q_head_num,
                                           int64_t q_seq_len, int64_t kv_seq_len) {
  if (std::find(kNeedCompressAttnMaskMode.begin(), kNeedCompressAttnMaskMode.end(), sparse_mode) !=
      kNeedCompressAttnMaskMode.end()) {
    CheckFlashAttentionScoreInputShape(
      attn_mask, {kInputFlashAttentionScoreAttnMaskCompressionDim, kInputFlashAttentionScoreAttnMaskCompressionDim},
      op_name, "attn_mask");
  } else {
    auto is_attn_mask_optional = (sparse_mode == kSparseDefaultMask || sparse_mode == kSparseAllMask);
    CheckFlashAttentionScoreInputShape(attn_mask,
                                       {{batch_size, q_head_num, q_seq_len, kv_seq_len},
                                        {batch_size, 1, q_seq_len, kv_seq_len},
                                        {1, 1, q_seq_len, kv_seq_len},
                                        {q_seq_len, kv_seq_len}},
                                       op_name, "attn_mask", is_attn_mask_optional);
  }
}

void CheckFlashAttentionScorePrefix(const InferInfoPtr &prefix, const std::string &op_name, int64_t sparse_mode,
                                    int64_t batch_size) {
  if (sparse_mode == kSparsePrefix) {
    auto arr_opt = prefix->GetArrayValue<int64_t>();
    if (!arr_opt.has_value() || arr_opt->HasUnknownValue()) {
      MS_LOG(EXCEPTION) << "For [" << op_name << "], prefix list must be known and not None.";
    }
    auto vec = arr_opt->ToVector();
    if (SizeToLong(vec.size()) != batch_size) {
      MS_LOG(EXCEPTION) << "For [" << op_name << "], prefix list size should be equal to " << batch_size << ", but got "
                        << vec.size();
    }
  } else {
    if (!IsFlashAttentionScoreOptionalInputNotPass(prefix)) {
      MS_LOG(EXCEPTION) << op_name << ": 'prefix' must be None if sparse_mode is not " << kSparsePrefix;
    }
  }
}

void CheckFlashAttentionScoreSparseMode(const PrimitivePtr &primitive, const InferInfoPtrList &input_infos,
                                        const std::vector<int64_t> &shape_info, int64_t q_head_num) {
  auto op_name = primitive->name();
  int64_t batch_size = shape_info[kIndex0];
  int64_t q_seq_len = shape_info[kIndex1];
  int64_t kv_seq_len = shape_info[kIndex2];
  auto sparse_mode_opt = input_infos[kFlashAttentionScoreInputSparseModeIndex]->GetScalarValue<int64_t>();
  if (sparse_mode_opt.has_value()) {
    auto sparse_mode = sparse_mode_opt.value();

    bool enable_ring_attention = false;
    if (primitive->HasAttr(kEnableRingAttention)) {
      auto enable_ring_attention_valueptr = primitive->GetAttr(kEnableRingAttention);
      if (enable_ring_attention_valueptr->isa<BoolImm>()) {
        enable_ring_attention = enable_ring_attention_valueptr->cast<BoolImmPtr>()->value();
      } else {
        MS_LOG(EXCEPTION) << "enable_ring_attention should be bool";
      }
    }
    if (primitive->HasAttr(kEnableRASendRecv)) {
      auto enable_ra_sendrecv_valueptr = primitive->GetAttr(kEnableRASendRecv);
      if (!(enable_ra_sendrecv_valueptr->isa<BoolImm>())) {
        MS_LOG(EXCEPTION) << "enable_ra_send_recv should be bool";
      }
    }
    bool enable_flash_sp = false;
    if (primitive->HasAttr(kEnableFlashSP)) {
      auto enable_flash_sp_valueptr = primitive->GetAttr(kEnableFlashSP);
      if (enable_flash_sp_valueptr->isa<BoolImm>()) {
        enable_flash_sp = enable_flash_sp_valueptr->cast<BoolImmPtr>()->value();
      } else {
        MS_LOG(ERROR) << "enable_flash_sp should be bool";
      }
    }
    if ((!enable_ring_attention && !enable_flash_sp) ||
        !IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputAttnMaskIndex])) {
      CheckFlashAttentionScoreAttnMaskShape(input_infos[kFlashAttentionScoreInputAttnMaskIndex], op_name, sparse_mode,
                                            batch_size, q_head_num, q_seq_len, kv_seq_len);
    }
    CheckFlashAttentionScorePrefix(input_infos[kFlashAttentionScoreInputPrefixIndex], op_name, sparse_mode, batch_size);
  }
}

ShapeArray ConstructInferShape(const ShapeVector &softmax_shape, const ShapeVector &query_shape,
                               const ShapeVector &key_shape, const ShapeVector &value_shape,
                               std::optional<int64_t> input_layout = std::nullopt) {
  auto output_shape = query_shape;
  if (input_layout.has_value() && !IsDynamicRank(query_shape) && !IsDynamicRank(key_shape) &&
      !IsDynamicRank(value_shape)) {
    auto input_layout_pair = layoutMap.find(input_layout.value());
    if (input_layout_pair == layoutMap.end()) {
      MS_LOG(EXCEPTION) << "FlashAttentionScore: unsupported layout: " << input_layout.value();
    }
    const std::string &input_layout_str = input_layout_pair->second;
    if (input_layout_str.find("D") != std::string::npos) {
      auto head_dim_index = input_layout_str.find("D");
      auto value_head_dim = value_shape.at(head_dim_index);
      output_shape.at(head_dim_index) = value_head_dim;
    } else {
      auto hidden_dim_index = input_layout_str.find("H");
      if (hidden_dim_index == std::string::npos) {
        MS_LOG(EXCEPTION) << "FlashAttentionScore: cannot find the head_dim or hidden dimension from layout "
                          << input_layout_str;
      }
      auto query_hidden_size = query_shape.at(hidden_dim_index);
      auto key_hidden_size = key_shape.at(hidden_dim_index);
      auto value_hidden_size = value_shape.at(hidden_dim_index);
      auto output_hidden_size = query_hidden_size / key_hidden_size * value_hidden_size;
      output_shape.at(hidden_dim_index) = output_hidden_size < 0 ? abstract::Shape::kShapeDimAny : output_hidden_size;
    }
  }
  return ShapeArray{softmax_shape, softmax_shape, ShapeVector{1}, output_shape};
}

std::vector<int64_t> GetFASInfoFromInputLayout(int64_t input_layout, int64_t q_head_num, const std::string &op_name,
                                               const ShapeVector &query_shape, const ShapeVector &key_shape,
                                               const ShapeVector &value_shape) {
  int64_t batch_size = -1;
  int64_t q_seq_len = -1;
  int64_t kv_seq_len = -1;
  int64_t kv_head_num = -1;
  if (query_shape.size() != key_shape.size() || query_shape.size() != value_shape.size()) {
    MS_LOG(EXCEPTION) << op_name << ": The rank among 'query', 'key' and 'value' must be the same, but got "
                      << query_shape.size() << ", " << key_shape.size() << " and " << value_shape.size();
  }
  if (input_layout == FASInputLayoutMode::BSH) {
    if (query_shape.size() != kInputFlashAttentionScoreQueryBSHRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFlashAttentionScoreQueryBSHRank
                        << ", but got " << query_shape.size() << " and " << key_shape.size();
    }
    batch_size = query_shape[0];
    q_seq_len = query_shape[1];
    auto q_hidden_size = query_shape[2];
    if (q_hidden_size % q_head_num != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << q_hidden_size
                        << " and " << q_head_num;
    }
    int64_t head_size = q_hidden_size / q_head_num;
    kv_seq_len = key_shape[kIndex1];
    kv_head_num = key_shape[kIndex2] / head_size;
  } else if (input_layout == FASInputLayoutMode::BNSD) {
    if (query_shape.size() != kInputFlashAttentionScoreQueryBNSDRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFlashAttentionScoreQueryBNSDRank
                        << ", but got " << query_shape.size();
    }
    batch_size = query_shape[kIndex0];
    if (q_head_num != query_shape[kIndex1]) {
      MS_LOG(EXCEPTION) << op_name << ": query_shape[1] must be equal to attribute 'head_num', but got "
                        << query_shape[1] << " and " << q_head_num;
    }
    q_seq_len = query_shape[kIndex2];
    kv_seq_len = key_shape[kIndex2];
    kv_head_num = key_shape[kIndex1];
  } else if (input_layout == FASInputLayoutMode::SBH) {
    if (query_shape.size() != kInputFlashAttentionScoreQuerySBHRank || key_shape.size() != query_shape.size()) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' and 'key' must be "
                        << kInputFlashAttentionScoreQuerySBHRank << ", but got " << query_shape.size() << " and "
                        << key_shape.size();
    }
    batch_size = query_shape[1];
    q_seq_len = query_shape[0];
    auto q_hidden_size = query_shape[2];
    if (q_hidden_size % q_head_num != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << q_hidden_size
                        << " and " << q_head_num;
    }
    int64_t head_size = q_hidden_size / q_head_num;
    kv_seq_len = key_shape[kIndex0];
    kv_head_num = key_shape[kIndex2] / head_size;
  } else if (input_layout == FASInputLayoutMode::BSND) {
    if (query_shape.size() != kInputFlashAttentionScoreQueryBSNDRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFlashAttentionScoreQueryBSNDRank
                        << ", but got " << query_shape.size();
    }
    batch_size = query_shape[kIndex0];
    if (q_head_num != query_shape[kIndex2]) {
      MS_LOG(EXCEPTION) << op_name << ": query_shape[2] must be equal to attribute 'head_num', but got "
                        << query_shape[kIndex2] << " and " << q_head_num;
    }
    q_seq_len = query_shape[kIndex1];
    kv_seq_len = key_shape[kIndex1];
    kv_head_num = key_shape[kIndex2];
  } else {
    MS_LOG(EXCEPTION) << op_name << " support input layout: BSH, BNSD, SBH, BSND, TND.";
  }
  if (q_head_num % kv_head_num != 0) {
    MS_LOG(EXCEPTION) << op_name << ": The head num of 'key' must be a factor of the head num of 'query', but got "
                      << kv_head_num << " and " << q_head_num;
  }
  return std::vector<int64_t>{batch_size, q_seq_len, kv_seq_len};
}

ShapeArray FlashAttentionScoreFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                   const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto query_shape = input_infos[kFlashAttentionScoreInputQueryIndex]->GetShape();
  auto key_shape = input_infos[kFlashAttentionScoreInputKeyIndex]->GetShape();
  auto value_shape = input_infos[kFlashAttentionScoreInputValueIndex]->GetShape();
  ShapeVector dyn_rank{abstract::Shape::kShapeRankAny};
  if (IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputLayoutIndex])) {
    return ConstructInferShape(dyn_rank, query_shape, key_shape, value_shape);
  }
  auto input_layout_opt = input_infos[kFlashAttentionScoreInputLayoutIndex]->GetScalarValue<int64_t>();
  if (!input_layout_opt.has_value()) {
    return ConstructInferShape(dyn_rank, query_shape, key_shape, value_shape);
  }

  bool head_num_no_value = false;
  std::optional<int64_t> head_num_opt_cached;
  if (IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputHeadNumIndex])) {
    head_num_no_value = true;
  } else {
    head_num_opt_cached = input_infos[kFlashAttentionScoreInputHeadNumIndex]->GetScalarValue<int64_t>();
    if (!head_num_opt_cached.has_value()) {
      head_num_no_value = true;
    }
  }

  auto input_layout = input_layout_opt.value();
  if (input_layout == FASInputLayoutMode::TND || input_layout == FASInputLayoutMode::TH) {
    if (IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputActualSeqQlenIndex]) ||
        IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputActualSeqKVlenIndex])) {
      MS_LOG(EXCEPTION) << op_name << ": actual_seq_qlen and actual_seq_kvlen should be not none.";
    }

    if (input_layout == FASInputLayoutMode::TND) {
      if (IsDynamicRank(query_shape)) {
        return ConstructInferShape(
          ShapeVector{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny, kFlashAttentionScoreSoftmaxLastDim},
          query_shape, key_shape, value_shape, input_layout_opt);
      }
      return ConstructInferShape(ShapeVector{query_shape[0], query_shape[1], kFlashAttentionScoreSoftmaxLastDim},
                                 query_shape, key_shape, value_shape, input_layout_opt);
    } else {
      if (IsDynamicRank(query_shape)) {
        return ConstructInferShape(ShapeVector{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny},
                                   query_shape, key_shape, value_shape, input_layout_opt);
      }
      if (!head_num_opt_cached.has_value()) {
        MS_LOG(EXCEPTION) << op_name << ": 'head_num' must be provided when input layout is TH.";
      }
      int64_t q_head_num = head_num_opt_cached.value();
      q_head_num *= static_cast<int64_t>(kFlashAttentionScoreSoftmaxLastDim);
      return ConstructInferShape(ShapeVector{query_shape[0], q_head_num}, query_shape, key_shape, value_shape,
                                 input_layout_opt);
    }
  }

  if (IsDynamicRank(query_shape)) {
    return ConstructInferShape(ShapeVector{abstract::Shape::kShapeDimAny, abstract::Shape::kShapeDimAny,
                                           abstract::Shape::kShapeDimAny, kFlashAttentionScoreSoftmaxLastDim},
                               query_shape, key_shape, value_shape, input_layout_opt);
  }

  size_t seq_index = kIndex1, batch_index = kIndex0;
  if (input_layout == FASInputLayoutMode::SBH) {
    seq_index = kIndex0;
    batch_index = kIndex1;
  } else if (input_layout == FASInputLayoutMode::BNSD) {
    seq_index = kIndex2;
  }
  if (head_num_no_value) {
    return ConstructInferShape(ShapeVector{query_shape[batch_index], abstract::Shape::kShapeDimAny,
                                           query_shape[seq_index], kFlashAttentionScoreSoftmaxLastDim},
                               query_shape, key_shape, value_shape, input_layout_opt);
  }

  auto q_head_num = head_num_opt_cached.value();
  if (IsDynamicShape(query_shape) || IsDynamic(key_shape)) {
    return ConstructInferShape(
      ShapeVector{query_shape[batch_index], q_head_num, query_shape[seq_index], kFlashAttentionScoreSoftmaxLastDim},
      query_shape, key_shape, value_shape, input_layout_opt);
  }

  auto shape_info = GetFASInfoFromInputLayout(input_layout, q_head_num, op_name, query_shape, key_shape, value_shape);
  int64_t batch_size = shape_info[kIndex0];
  int64_t q_seq_len = shape_info[kIndex1];
  int64_t kv_seq_len = shape_info[kIndex2];

  CheckFlashAttentionScoreInputShape(input_infos[kFlashAttentionScoreInputRealShiftIndex],
                                     {{batch_size, q_head_num, q_seq_len, kv_seq_len},
                                      {1, q_head_num, q_seq_len, kv_seq_len},
                                      {batch_size, q_head_num, kFAGRealShiftCompressionDim, kv_seq_len},
                                      {1, q_head_num, kFAGRealShiftCompressionDim, kv_seq_len}},
                                     op_name, "real_shift", true);
  CheckFlashAttentionScoreInputShape(input_infos[kFlashAttentionScoreInputDropMaskIndex],
                                     {batch_size, q_head_num, q_seq_len, kv_seq_len / 8}, op_name, "drop_mask", true);
  CheckFlashAttentionScoreSparseMode(primitive, input_infos, shape_info, q_head_num);

  return ConstructInferShape(ShapeVector{batch_size, q_head_num, q_seq_len, kFlashAttentionScoreSoftmaxLastDim},
                             query_shape, key_shape, value_shape, input_layout_opt);
}

std::vector<TypeId> FlashAttentionScoreFuncImpl::InferType(const PrimitivePtr &prim,
                                                           const InferInfoPtrList &input_infos) const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  const bool enable_infer_boost = ms_context->IsEnableInferBoost();
  auto op_name = prim->name();

  // 1) Q/K/V must have same dtype
  const auto q_type = input_infos[kFlashAttentionScoreInputQueryIndex]->GetType();
  const auto k_type = input_infos[kFlashAttentionScoreInputKeyIndex]->GetType();
  const auto v_type = input_infos[kFlashAttentionScoreInputValueIndex]->GetType();
  std::vector<TypeId> qkv_types{q_type, k_type, v_type};
  CheckAndConvertUtils::CheckTypeIdsSame("query/key/value", qkv_types, op_name);

  // 2) Validate QKV dtype set depending on infer boost
  std::set<TypeId> valid_qkv_types = {kNumberTypeFloat16, kNumberTypeBFloat16};
  if (!enable_infer_boost) {
    (void)valid_qkv_types.emplace(kNumberTypeFloat32);
  }
  CheckAndConvertUtils::CheckTypeIdValid("query", q_type, valid_qkv_types, op_name);

  // 3) padding_mask must be None currently
  EnsurePaddingMaskNone(input_infos[kFlashAttentionScoreInputPaddingMaskIndex], op_name);

  // 4) real_shift, if provided, must share dtype with QKV
  if (!IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputRealShiftIndex])) {
    const auto real_shift_type = input_infos[kFlashAttentionScoreInputRealShiftIndex]->GetType();
    std::vector<TypeId> types{q_type, real_shift_type};
    CheckAndConvertUtils::CheckTypeIdsSame("query/real_shift", types, op_name);
  }

  // 5) attn_mask dtype must be valid when provided
  if (!IsFlashAttentionScoreOptionalInputNotPass(input_infos[kFlashAttentionScoreInputAttnMaskIndex])) {
    const auto attn_mask_type = input_infos[kFlashAttentionScoreInputAttnMaskIndex]->GetType();
    std::set<TypeId> valid_attn_types = {kNumberTypeUInt8, kNumberTypeBool};
    if (enable_infer_boost) {
      (void)valid_attn_types.emplace(kNumberTypeFloat16);
      (void)valid_attn_types.emplace(kNumberTypeBFloat16);
    }
    CheckAndConvertUtils::CheckTypeIdValid("attn_mask", attn_mask_type, valid_attn_types, op_name);
  }

  // 6) keep_prob/drop_mask rule
  ValidateKeepProbAndDropMask(input_infos, op_name);

  std::vector<TypeId> outs(kFlashAttentionScoreOutputsNum);
  outs[kFlashAttentionScoreOutputSoftmaxMaxIndex] = kNumberTypeFloat32;
  outs[kFlashAttentionScoreOutputSoftmaxSumIndex] = kNumberTypeFloat32;
  outs[kFlashAttentionScoreOutputSoftmaxOutIndex] = q_type;
  outs[kFlashAttentionScoreOutputAttentionOutIndex] = q_type;
  return outs;
}
}  // namespace ops
}  // namespace mindspore
