/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "infer/ops_func_impl/flash_attention_score_grad.h"

#include <string>
#include <map>
#include <memory>

#include "abstract/ops/primitive_infer_map.h"
#include "mindspore/ops/op_def/nn_ops.h"
#include "utils/check_convert_utils.h"
#include "mindspore/ops/op_def/op_enum.h"
#include "mindapi/helper.h"
#include "mindspore/ops/ops_utils/op_utils.h"
#include "ops_utils/op_constants.h"

namespace mindspore {
namespace ops {
constexpr size_t kFASGradSoftmaxLastDim = 8;
constexpr size_t kInputFASGradQueryBSHRank = 3;
constexpr size_t kInputFASGradQuerySBHRank = 3;
constexpr size_t kInputFASGradQueryTNDRank = 3;
constexpr size_t kInputFASGradQueryBNSDRank = 4;
constexpr size_t kInputFASGradQueryBSNDRank = 4;
constexpr size_t kInputFASGradAttnMaskCompressionDim = 2048;
constexpr size_t kFARealShiftCompressionDim = 1024;
// None indicates that the optional input is not passed
bool IsFlashAttentionScoreGradOptionalInputNotPass(const InferInfoPtr &input) { return input->IsNone(); }

static inline void ValidateGradAttnMaskType(const InferInfoPtr &attn_mask, const std::string &op_name) {
  if (IsFlashAttentionScoreGradOptionalInputNotPass(attn_mask)) {
    return;
  }
  auto attn_mask_type = attn_mask->GetType();
  if (!(attn_mask_type == kNumberTypeUInt8 || attn_mask_type == kNumberTypeBool)) {
    MS_LOG(EXCEPTION) << op_name << ": invalid dtype for attn_mask.";
  }
}

static inline void EnsureGradPaddingMaskNone(const InferInfoPtr &padding_mask, const std::string &op_name) {
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(padding_mask)) {
    MS_LOG(EXCEPTION) << op_name << ": 'padding_mask' must be None currently.";
  }
}

static inline void ValidateGradKeepProbAndDropMask(const InferInfoPtrList &input_infos, const std::string &op_name) {
  auto keep_prob_opt = input_infos[kFASGradInputKeepProbIndex]->GetScalarValue<float>();
  if (!keep_prob_opt.has_value()) {
    return;
  }
  auto keep_prob = keep_prob_opt.value();
  if (keep_prob > 1 || keep_prob < 0) {
    MS_LOG(EXCEPTION) << op_name << ": attribute `keep_prob` must be a floating point number in [0, 1], but got "
                      << keep_prob;
  }
  if (common::IsFloatEqual(keep_prob, 1.0)) {
    if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_infos[kFASGradInputDropMaskIndex])) {
      MS_LOG(EXCEPTION) << op_name << ": 'drop_mask' must be None when keep_prob is 1.0.";
    }
    return;
  }
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_infos[kFASGradInputDropMaskIndex])) {
    auto drop_mask_type = input_infos[kFASGradInputDropMaskIndex]->GetType();
    if (drop_mask_type != kNumberTypeUInt8) {
      MS_LOG(EXCEPTION) << op_name << ": 'drop_mask' must be uint8 when keep_prob in (0, 1).";
    }
  }
}

void CheckFlashAttentionScoreGradInputShape(const InferInfoPtr &input, const ShapeVector &expect_shape,
                                            const std::string &op_name, const std::string &input_name,
                                            bool optional = false) {
  if (IsFlashAttentionScoreGradOptionalInputNotPass(input) && optional) {
    return;
  }
  auto input_shape = input->GetShape();
  if (input_shape != expect_shape) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input `" << input_name << "' must be " << expect_shape
                      << ", but got shape is " << input_shape;
  }
}

void CheckFlashAttentionScoreGradInputShape(const InferInfoPtr &input,
                                            const std::vector<ShapeVector> &expect_shape_list,
                                            const std::string &op_name, const std::string &input_name,
                                            bool optional = false) {
  if (IsFlashAttentionScoreGradOptionalInputNotPass(input) && optional) {
    return;
  }
  auto input_shape = input->GetShape();
  if (std::all_of(expect_shape_list.begin(), expect_shape_list.end(),
                  [&input_shape](const ShapeVector &expect_shape) { return input_shape != expect_shape; })) {
    MS_LOG(EXCEPTION) << op_name << ": The shape of input `" << input_name << "' must be one of " << expect_shape_list
                      << ", but got shape is " << input_shape;
  }
}

void CheckFlashAttentionScoreGradAttnMaskShape(const InferInfoPtr &attn_mask, const std::string &op_name,
                                               int64_t sparse_mode, int64_t batch_size, int64_t q_head_num,
                                               int64_t q_seq_len, int64_t kv_seq_len) {
  const std::vector<int64_t> need_compress_attn_mask_mode = {kFAGSparseLeftUpCausal, kFAGSparseRightDownCausal,
                                                             kFAGSparseBand};
  if (std::find(need_compress_attn_mask_mode.begin(), need_compress_attn_mask_mode.end(), sparse_mode) !=
      need_compress_attn_mask_mode.end()) {
    CheckFlashAttentionScoreGradInputShape(
      attn_mask, {kInputFASGradAttnMaskCompressionDim, kInputFASGradAttnMaskCompressionDim}, op_name, "attn_mask");
  } else {
    auto is_attn_mask_optional = (sparse_mode == kFAGSparseDefaultMask || sparse_mode == kFAGSparseAllMask);
    CheckFlashAttentionScoreGradInputShape(attn_mask,
                                           {{batch_size, q_head_num, q_seq_len, kv_seq_len},
                                            {batch_size, 1, q_seq_len, kv_seq_len},
                                            {1, 1, q_seq_len, kv_seq_len},
                                            {q_seq_len, kv_seq_len}},
                                           op_name, "attn_mask", is_attn_mask_optional);
  }
}

void CheckFlashAttentionScoreGradPrefix(const InferInfoPtr &prefix, const std::string &op_name, int64_t sparse_mode,
                                        int64_t batch_size) {
  if (sparse_mode == kFAGSparsePrefix) {
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
    if (!IsFlashAttentionScoreGradOptionalInputNotPass(prefix)) {
      MS_LOG(EXCEPTION) << op_name << ": 'prefix' must be None if sparse_mode is not " << kFAGSparsePrefix;
    }
  }
}

std::vector<int64_t> GetFASGradInfoFromInputLayout(int64_t input_layout, int64_t q_head_num, const std::string &op_name,
                                                   const ShapeVector &query_shape, const ShapeVector &key_shape) {
  int64_t batch_size = -1;
  int64_t q_seq_len = -1;
  int64_t kv_seq_len = -1;
  int64_t kv_head_num = -1;
  if (input_layout == FASInputLayoutMode::BSH) {
    if (query_shape.size() != kInputFASGradQueryBSHRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of input `query` must be " << kInputFASGradQueryBSHRank
                        << ", but got " << query_shape.size();
    }
    batch_size = query_shape[kIndex0];
    q_seq_len = query_shape[kIndex1];
    auto q_hidden_size = query_shape[kIndex2];
    if (q_hidden_size % q_head_num != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << q_hidden_size
                        << " and " << q_head_num;
    }
    int64_t head_size = q_hidden_size / q_head_num;
    kv_seq_len = key_shape[kIndex1];
    kv_head_num = key_shape[kIndex2] / head_size;
  } else if (input_layout == FASInputLayoutMode::BNSD) {
    if (query_shape.size() != kInputFASGradQueryBNSDRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFASGradQueryBNSDRank << ", but got "
                        << query_shape.size();
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
    if (query_shape.size() != kInputFASGradQuerySBHRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of input `query` must be " << kInputFASGradQuerySBHRank
                        << ", but got " << query_shape.size();
    }
    batch_size = query_shape[kIndex1];
    q_seq_len = query_shape[kIndex0];
    auto q_hidden_size = query_shape[kIndex2];
    if (q_hidden_size % q_head_num != 0) {
      MS_LOG(EXCEPTION) << op_name << ": 'hidden_size` must be divisible by `head_num`, but got " << q_hidden_size
                        << " and " << q_head_num;
    }
    int64_t head_size = q_hidden_size / q_head_num;
    kv_seq_len = key_shape[kIndex0];
    kv_head_num = key_shape[kIndex2] / head_size;
  } else if (input_layout == FASInputLayoutMode::BSND) {
    if (query_shape.size() != kInputFASGradQueryBSNDRank) {
      MS_LOG(EXCEPTION) << op_name << ": The rank of 'query' must be " << kInputFASGradQueryBSNDRank << ", but got "
                        << query_shape.size();
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
    MS_LOG(EXCEPTION) << op_name << ": The head num of key must be a factor of the head num of query, but got "
                      << kv_head_num << " and " << q_head_num;
  }
  return std::vector<int64_t>{batch_size, q_seq_len, kv_seq_len};
}

ShapeArray FlashAttentionScoreGradFuncImpl::InferShape(const PrimitivePtr &primitive,
                                                       const InferInfoPtrList &input_infos) const {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto query_shape = input_infos[kFASGradInputQueryIndex]->GetShape();
  auto key_shape = input_infos[kFASGradInputKeyIndex]->GetShape();

  ShapeArray out_shapes(kFASGradOutputsNum);
  out_shapes[kFASGradOutputDqIndex] = query_shape;
  out_shapes[kFASGradOutputDkIndex] = key_shape;
  auto value_shape = input_infos[kFASGradInputValueIndex]->GetShape();
  out_shapes[kFASGradOutputDvIndex] = value_shape;
  ShapeVector pse_shape{0};
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_infos[kFASGradInputPseShiftIndex])) {
    pse_shape = input_infos[kFASGradInputPseShiftIndex]->GetShape();
  }
  out_shapes[kFASGradOutputDpseIndex] = pse_shape;

  auto input_layout_opt = input_infos[kFASGradInputLayoutIndex]->GetScalarValue<int64_t>();
  if (!input_layout_opt.has_value() || IsDynamic(query_shape) || IsDynamic(key_shape)) {
    return out_shapes;
  }
  auto input_layout = input_layout_opt.value();
  if (input_layout == FASInputLayoutMode::TND) {
    if (IsFlashAttentionScoreGradOptionalInputNotPass(input_infos[kFASGradInputActualSeqQlenIndex]) ||
        IsFlashAttentionScoreGradOptionalInputNotPass(input_infos[kFASGradInputActualSeqKVlenIndex])) {
      MS_LOG(EXCEPTION) << op_name << ": actual_seq_qlen and actual_seq_kvlen should be not none.";
    }
    return out_shapes;
  }

  auto q_head_num_opt = input_infos[kFASGradInputHeadNumIndex]->GetScalarValue<int64_t>();
  if (q_head_num_opt.has_value()) {
    // check shape
    auto q_head_num = q_head_num_opt.value();
    auto shape_info = GetFASGradInfoFromInputLayout(input_layout, q_head_num, op_name, query_shape, key_shape);
    int64_t batch_size = shape_info[kIndex0];
    int64_t q_seq_len = shape_info[kIndex1];
    int64_t kv_seq_len = shape_info[kIndex2];

    CheckFlashAttentionScoreGradInputShape(input_infos[kFASGradInputPseShiftIndex],
                                           {{batch_size, q_head_num, q_seq_len, kv_seq_len},
                                            {1, q_head_num, q_seq_len, kv_seq_len},
                                            {batch_size, q_head_num, kFARealShiftCompressionDim, kv_seq_len},
                                            {1, q_head_num, kFARealShiftCompressionDim, kv_seq_len}},
                                           op_name, "pse_shift", true);
    CheckFlashAttentionScoreGradInputShape(input_infos[kFASGradInputDropMaskIndex],
                                           {batch_size, q_head_num, q_seq_len, kv_seq_len / 8}, op_name, "drop_mask",
                                           true);

    auto sparse_mode_opt = input_infos[kFASGradInputSparseModeIndex]->GetScalarValue<int64_t>();
    if (sparse_mode_opt.has_value()) {
      auto sparse_mode = sparse_mode_opt.value();
      CheckFlashAttentionScoreGradAttnMaskShape(input_infos[kFASGradInputAttnMaskIndex], op_name, sparse_mode,
                                                batch_size, q_head_num, q_seq_len, kv_seq_len);
      CheckFlashAttentionScoreGradPrefix(input_infos[kFASGradInputPrefixIndex], op_name, sparse_mode, batch_size);
    }

    CheckFlashAttentionScoreGradInputShape(input_infos[kFASGradInputSoftmaxMaxIndex],
                                           {batch_size, q_head_num, q_seq_len, kFASGradSoftmaxLastDim}, op_name,
                                           "softmax_max");
    CheckFlashAttentionScoreGradInputShape(input_infos[kFASGradInputSoftmaxSumIndex],
                                           {batch_size, q_head_num, q_seq_len, kFASGradSoftmaxLastDim}, op_name,
                                           "softmax_sum");
    CheckFlashAttentionScoreGradInputShape(input_infos[kFASGradInputSoftmaxOutIndex], ShapeVector{1}, op_name,
                                           "softmax_out", true);
  }
  return out_shapes;
}

std::vector<TypeId> FlashAttentionScoreGradFuncImpl::InferType(const PrimitivePtr &prim,
                                                               const InferInfoPtrList &input_infos) const {
  auto op_name = prim->name();
  ValidateGradAttnMaskType(input_infos[kFASGradInputAttnMaskIndex], op_name);
  EnsureGradPaddingMaskNone(input_infos[kFASGradInputPaddingMaskIndex], op_name);
  // 1) (query,key,value,dy) must have same dtype
  const auto q_type = input_infos[kFASGradInputQueryIndex]->GetType();
  const auto k_type = input_infos[kFASGradInputKeyIndex]->GetType();
  const auto v_type = input_infos[kFASGradInputValueIndex]->GetType();
  const auto dy_type = input_infos[kFASGradInputDyIndex]->GetType();
  std::vector<TypeId> qkvd_types{q_type, k_type, v_type, dy_type};
  CheckAndConvertUtils::CheckTypeIdsSame("query/key/value/dy", qkvd_types, op_name);
  // 2) qkv/dy dtype must be valid
  CheckAndConvertUtils::CheckTypeIdValid("query", q_type, {kNumberTypeFloat16, kNumberTypeBFloat16, kNumberTypeFloat32},
                                         op_name);
  // Ensure optional tensors keep dtype consistency with q_type
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_infos[kFASGradInputPseShiftIndex])) {
    auto pse_type = input_infos[kFASGradInputPseShiftIndex]->GetType();
    std::vector<TypeId> types{q_type, pse_type};
    CheckAndConvertUtils::CheckTypeIdsSame("pse_shift", types, op_name);
  }
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_infos[kFASGradInputAttentionInIndex])) {
    auto attention_in_type = input_infos[kFASGradInputAttentionInIndex]->GetType();
    std::vector<TypeId> types{q_type, attention_in_type};
    CheckAndConvertUtils::CheckTypeIdsSame("attention_in", types, op_name);
  }
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_infos[kFASGradInputSoftmaxOutIndex])) {
    auto softmax_out_type = input_infos[kFASGradInputSoftmaxOutIndex]->GetType();
    std::vector<TypeId> types{q_type, softmax_out_type};
    CheckAndConvertUtils::CheckTypeIdsSame("softmax_out", types, op_name);
  }
  // Stats tensors must be float32 when provided
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_infos[kFASGradInputSoftmaxMaxIndex])) {
    auto softmax_max_type = input_infos[kFASGradInputSoftmaxMaxIndex]->GetType();
    CheckAndConvertUtils::CheckTypeIdValid("softmax_max", softmax_max_type, {kNumberTypeFloat32}, op_name);
  }
  if (!IsFlashAttentionScoreGradOptionalInputNotPass(input_infos[kFASGradInputSoftmaxSumIndex])) {
    auto softmax_sum_type = input_infos[kFASGradInputSoftmaxSumIndex]->GetType();
    CheckAndConvertUtils::CheckTypeIdValid("softmax_sum", softmax_sum_type, {kNumberTypeFloat32}, op_name);
  }
  ValidateGradKeepProbAndDropMask(input_infos, op_name);

  std::vector<TypeId> outs(kFASGradOutputsNum);
  outs[kFASGradOutputDqIndex] = q_type;
  outs[kFASGradOutputDkIndex] = q_type;
  outs[kFASGradOutputDvIndex] = q_type;
  outs[kFASGradOutputDpseIndex] = q_type;
  return outs;
}

}  // namespace ops
}  // namespace mindspore
