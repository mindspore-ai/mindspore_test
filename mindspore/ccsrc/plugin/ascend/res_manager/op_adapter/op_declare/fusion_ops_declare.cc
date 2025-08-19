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

#include "plugin/ascend/res_manager/op_adapter/op_declare/fusion_ops_declare.h"
#include <vector>
#include <string>
#include "mindspore/ops/infer/ops_func_impl/all_gather_matmul.h"
#include "mindspore/ops/infer/ops_func_impl/matmul_reduce_scatter.h"

namespace mindspore::device::ascend {
// PromptFlashAttention
INPUT_MAP(PromptFlashAttention) = {
  {kIndex1, INPUT_DESC(query)},
  {kIndex2, INPUT_DESC(key)},
  {kIndex3, INPUT_DESC(value)},
  {kIndex4, INPUT_DESC(atten_mask)},             // optional input
  {kIndex5, INPUT_DESC(actual_seq_lengths)},     // optional input
  {kIndex6, INPUT_DESC(actual_seq_lengths_kv)},  // optional input
  {kIndex7, INPUT_DESC(pse_shift)},              // optional input
  {kIndex8, INPUT_DESC(deq_scale1)},             // optional input
  {kIndex9, INPUT_DESC(quant_scale1)},           // optional input
  {kIndex10, INPUT_DESC(deq_scale2)},            // optional input
  {kIndex11, INPUT_DESC(quant_scale2)},          // optional input
  {kIndex12, INPUT_DESC(quant_offset2)}          // optional input
};
ATTR_MAP(PromptFlashAttention) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(PromptFlashAttention) = {{kIndex13, ATTR_DESC(num_heads, AnyTraits<int64_t>())},
                                        {kIndex14, ATTR_DESC(scale_value, AnyTraits<float>())},
                                        {kIndex15, ATTR_DESC(pre_tokens, AnyTraits<int64_t>())},
                                        {kIndex16, ATTR_DESC(next_tokens, AnyTraits<int64_t>())},
                                        {kIndex17, ATTR_DESC(input_layout, AnyTraits<FASInputLayoutMode>())},
                                        {kIndex18, ATTR_DESC(num_key_value_heads, AnyTraits<int64_t>())},
                                        {kIndex19, ATTR_DESC(sparse_mode, AnyTraits<int64_t>())},
                                        {kIndex20, ATTR_DESC(inner_precise, AnyTraits<int64_t>())}};
OUTPUT_MAP(PromptFlashAttention) = {{kIndex0, OUTPUT_DESC(attention_out)}};
REG_ADPT_DESC(PromptFlashAttention, "PromptFlashAttention", ADPT_DESC(PromptFlashAttention))

// IncreFlashAttention
INPUT_MAP(IncreFlashAttention) = {{1, INPUT_DESC(query)},
                                  {4, INPUT_DESC(atten_mask)},
                                  {5, INPUT_DESC(actual_seq_lengths)},
                                  {6, INPUT_DESC(pse_shift)},
                                  {7, INPUT_DESC(dequant_scale1)},
                                  {8, INPUT_DESC(quant_scale1)},
                                  {9, INPUT_DESC(dequant_scale2)},
                                  {10, INPUT_DESC(quant_scale2)},
                                  {11, INPUT_DESC(quant_offset2)},
                                  {12, INPUT_DESC(antiquant_scale)},
                                  {13, INPUT_DESC(antiquant_offset)},
                                  {14, INPUT_DESC(block_table)},
                                  {15, INPUT_DESC(kv_padding_size)}};
DYN_INPUT_MAP(IncreFlashAttention) = {{2, DYN_INPUT_DESC(key)}, {3, DYN_INPUT_DESC(value)}};
ATTR_MAP(IncreFlashAttention) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(IncreFlashAttention) = {
  {16, ATTR_DESC(num_heads, AnyTraits<int64_t>())},  {17, ATTR_DESC(input_layout, AnyTraits<FASInputLayoutMode>())},
  {18, ATTR_DESC(scale_value, AnyTraits<float>())},  {19, ATTR_DESC(num_key_value_heads, AnyTraits<int64_t>())},
  {20, ATTR_DESC(block_size, AnyTraits<int64_t>())}, {21, ATTR_DESC(inner_precise, AnyTraits<int64_t>())}};
OUTPUT_MAP(IncreFlashAttention) = {{0, OUTPUT_DESC(attention_out)}};
REG_ADPT_DESC(IncreFlashAttention, "IncreFlashAttention", ADPT_DESC(IncreFlashAttention))

// FlashAttentionScore
INPUT_MAP(FlashAttentionScore) = {{kIndex1, INPUT_DESC(query)},           {kIndex2, INPUT_DESC(key)},
                                  {kIndex3, INPUT_DESC(value)},           {kIndex4, INPUT_DESC(real_shift)},
                                  {kIndex5, INPUT_DESC(drop_mask)},       {kIndex6, INPUT_DESC(padding_mask)},
                                  {kIndex7, INPUT_DESC(atten_mask)},      {kIndex8, INPUT_DESC(prefix)},
                                  {kIndex9, INPUT_DESC(actual_seq_qlen)}, {kIndex10, INPUT_DESC(actual_seq_kvlen)}};
ATTR_MAP(FlashAttentionScore) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(FlashAttentionScore) = {
  {kIndex11, ATTR_DESC(head_num, AnyTraits<int64_t>())},
  {kIndex12, ATTR_DESC(keep_prob, AnyTraits<float>())},
  {kIndex13, ATTR_DESC(scale_value, AnyTraits<float>())},
  {kIndex14, ATTR_DESC(pre_tockens, AnyTraits<int64_t>())},
  {kIndex15, ATTR_DESC(next_tockens, AnyTraits<int64_t>())},
  {kIndex16, ATTR_DESC(inner_precise, AnyTraits<int64_t>())},
  {kIndex17, ATTR_DESC(input_layout, AnyTraits<FASInputLayoutMode>())},
  {kIndex18, ATTR_DESC(sparse_mode, AnyTraits<int64_t>())},
};
OUTPUT_MAP(FlashAttentionScore) = {{kIndex0, OUTPUT_DESC(softmax_max)},
                                   {kIndex1, OUTPUT_DESC(softmax_sum)},
                                   {kIndex2, OUTPUT_DESC(softmax_out)},
                                   {kIndex3, OUTPUT_DESC(attention_out)}};
REG_ADPT_DESC(FlashAttentionScore, kNameFlashAttentionScore, ADPT_DESC(FlashAttentionScore))

// FlashAttentionScoreGrad
INPUT_MAP(FlashAttentionScoreGrad) = {{kIndex1, INPUT_DESC(query)},
                                      {kIndex2, INPUT_DESC(key)},
                                      {kIndex3, INPUT_DESC(value)},
                                      {kIndex4, INPUT_DESC(dy)},
                                      {kIndex5, INPUT_DESC(pse_shift)},
                                      {kIndex6, INPUT_DESC(drop_mask)},
                                      {kIndex7, INPUT_DESC(padding_mask)},
                                      {kIndex8, INPUT_DESC(atten_mask)},
                                      {kIndex9, INPUT_DESC(softmax_max)},
                                      {kIndex10, INPUT_DESC(softmax_sum)},
                                      {kIndex11, INPUT_DESC(softmax_in)},
                                      {kIndex12, INPUT_DESC(attention_in)},
                                      {kIndex13, INPUT_DESC(prefix)},
                                      {kIndex14, INPUT_DESC(actual_seq_qlen)},
                                      {kIndex15, INPUT_DESC(actual_seq_kvlen)}};
ATTR_MAP(FlashAttentionScoreGrad) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(FlashAttentionScoreGrad) = {
  {kIndex16, ATTR_DESC(head_num, AnyTraits<int64_t>())},
  {kIndex17, ATTR_DESC(keep_prob, AnyTraits<float>())},
  {kIndex18, ATTR_DESC(scale_value, AnyTraits<float>())},
  {kIndex19, ATTR_DESC(pre_tockens, AnyTraits<int64_t>())},
  {kIndex20, ATTR_DESC(next_tockens, AnyTraits<int64_t>())},
  {kIndex21, ATTR_DESC(inner_precise, AnyTraits<int64_t>())},
  {kIndex22, ATTR_DESC(input_layout, AnyTraits<FASInputLayoutMode>())},
  {kIndex23, ATTR_DESC(sparse_mode, AnyTraits<int64_t>())},
};
OUTPUT_MAP(FlashAttentionScoreGrad) = {
  {kIndex0, OUTPUT_DESC(dq)}, {kIndex1, OUTPUT_DESC(dk)}, {kIndex2, OUTPUT_DESC(dv)}, {kIndex3, OUTPUT_DESC(dpse)}};
REG_ADPT_DESC(FlashAttentionScoreGrad, kNameFlashAttentionScoreGrad, ADPT_DESC(FlashAttentionScoreGrad))

// FusedInferAttentionScore
INPUT_MAP(FusedInferAttentionScore) = {{kIndex1, INPUT_DESC(query)},
                                       {kIndex4, INPUT_DESC(pse_shift)},
                                       {kIndex5, INPUT_DESC(atten_mask)},
                                       {kIndex6, INPUT_DESC(actual_seq_lengths)},
                                       {kIndex7, INPUT_DESC(actual_seq_lengths_kv)},
                                       {kIndex8, INPUT_DESC(dequant_scale1)},
                                       {kIndex9, INPUT_DESC(quant_scale1)},
                                       {kIndex10, INPUT_DESC(dequant_scale2)},
                                       {kIndex11, INPUT_DESC(quant_scale2)},
                                       {kIndex12, INPUT_DESC(quant_offset2)},
                                       {kIndex13, INPUT_DESC(antiquant_scale)},
                                       {kIndex14, INPUT_DESC(antiquant_offset)},
                                       {kIndex15, INPUT_DESC(block_table)},
                                       {kIndex16, INPUT_DESC(query_padding_size)},
                                       {kIndex17, INPUT_DESC(kv_padding_size)},
                                       {kIndex18, INPUT_DESC(key_antiquant_scale)},
                                       {kIndex19, INPUT_DESC(key_antiquant_offset)},
                                       {kIndex20, INPUT_DESC(value_antiquant_scale)},
                                       {kIndex21, INPUT_DESC(value_antiquant_offset)},
                                       {kIndex22, INPUT_DESC(key_shared_prefix)},
                                       {kIndex23, INPUT_DESC(value_shared_prefix)},
                                       {kIndex24, INPUT_DESC(actual_shared_prefix_len)}};
DYN_INPUT_MAP(FusedInferAttentionScore) = {{kIndex2, DYN_INPUT_DESC(key)}, {kIndex3, DYN_INPUT_DESC(value)}};
ATTR_MAP(FusedInferAttentionScore) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(FusedInferAttentionScore) = {
  {kIndex25, ATTR_DESC(num_heads, AnyTraits<int64_t>())},
  {kIndex26, ATTR_DESC(scale, AnyTraits<float>())},
  {kIndex27, ATTR_DESC(pre_tokens, AnyTraits<int64_t>())},
  {kIndex28, ATTR_DESC(next_tokens, AnyTraits<int64_t>())},
  {kIndex29, ATTR_DESC(input_layout, AnyTraits<FASInputLayoutMode>())},
  {kIndex30, ATTR_DESC(num_key_value_heads, AnyTraits<int64_t>())},
  {kIndex31, ATTR_DESC(sparse_mode, AnyTraits<int64_t>())},
  {kIndex32, ATTR_DESC(inner_precise, AnyTraits<int64_t>())},
  {kIndex33, ATTR_DESC(block_size, AnyTraits<int64_t>())},
  {kIndex34, ATTR_DESC(antiquant_mode, AnyTraits<int64_t>())},
  {kIndex35, ATTR_DESC(softmax_lse_flag, AnyTraits<bool>())},
  {kIndex36, ATTR_DESC(key_antiquant_mode, AnyTraits<int64_t>())},
  {kIndex37, ATTR_DESC(value_antiquant_mode, AnyTraits<int64_t>())},
};
OUTPUT_MAP(FusedInferAttentionScore) = {{kIndex0, OUTPUT_DESC(attention_out)}, {kIndex1, OUTPUT_DESC(softmax_lse)}};
REG_ADPT_DESC(FusedInferAttentionScore, "FusedInferAttentionScore", ADPT_DESC(FusedInferAttentionScore))

const std::vector<std::string> MatmulReduceScatterReduceOp = {"sum"};
INPUT_MAP(MatmulReduceScatter) = {{mindspore::ops::kMatmulReduceScatterInputInputIndex + 1, INPUT_DESC(x1)},
                                  {mindspore::ops::kMatmulReduceScatterInputX2Index + 1, INPUT_DESC(x2)},
                                  {mindspore::ops::kMatmulReduceScatterInputBiasIndex + 1, INPUT_DESC(bias)}};
INPUT_ATTR_MAP(MatmulReduceScatter) = {
  {mindspore::ops::kMatmulReduceScatterInputReduceOpIndex + 1,
   ATTR_DESC(reduce_op, AnyTraits<GEEnumToStr>(), MatmulReduceScatterReduceOp)},
  {mindspore::ops::kMatmulReduceScatterInputCommTurnIndex + 1, ATTR_DESC(comm_turn, AnyTraits<int64_t>())},
  {mindspore::ops::kMatmulReduceScatterInputTransInputIndex + 1, ATTR_DESC(is_trans_a, AnyTraits<bool>())},
  {mindspore::ops::kMatmulReduceScatterInputTransX2Index + 1, ATTR_DESC(is_trans_b, AnyTraits<bool>())}};
ATTR_MAP(MatmulReduceScatter) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MatmulReduceScatter) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MatmulReduceScatter, kNameMatmulReduceScatter, ADPT_DESC(MatmulReduceScatter))

INPUT_MAP(AllGatherMatmul) = {{mindspore::ops::kAllGatherMatmulInputInputIndex + 1, INPUT_DESC(x1)},
                              {mindspore::ops::kAllGatherMatmulInputX2Index + 1, INPUT_DESC(x2)},
                              {mindspore::ops::kAllGatherMatmulInputBiasIndex + 1, INPUT_DESC(bias)}};
INPUT_ATTR_MAP(AllGatherMatmul) = {
  {mindspore::ops::kAllGatherMatmulInputGatherIndexIndex + 1, ATTR_DESC(gather_index, AnyTraits<int64_t>())},
  {mindspore::ops::kAllGatherMatmulInputCommTurnIndex + 1, ATTR_DESC(comm_turn, AnyTraits<int64_t>())},
  {mindspore::ops::kAllGatherMatmulInputTransInputIndex + 1, ATTR_DESC(is_trans_a, AnyTraits<bool>())},
  {mindspore::ops::kAllGatherMatmulInputTransX2Index + 1, ATTR_DESC(is_trans_b, AnyTraits<bool>())}};
ATTR_MAP(AllGatherMatmul) = EMPTY_ATTR_MAP;
OUTPUT_MAP(AllGatherMatmul) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(gather_out)}};
REG_ADPT_DESC(AllGatherMatmul, kNameAllGatherMatmul, ADPT_DESC(AllGatherMatmul))

// GroupedMatmul
DYN_INPUT_MAP(GroupedMatmul) = {{1, DYN_INPUT_DESC(x)},
                                {2, DYN_INPUT_DESC(weight)},
                                {3, DYN_INPUT_DESC(bias)},
                                {4, DYN_INPUT_DESC(scale)},
                                {5, DYN_INPUT_DESC(offset)},
                                {6, DYN_INPUT_DESC(antiquant_scale)},
                                {7, DYN_INPUT_DESC(antiquant_offset)}};
INPUT_MAP(GroupedMatmul) = {{8, INPUT_DESC(group_list)}};
INPUT_ATTR_MAP(GroupedMatmul) = {{9, ATTR_DESC(split_item, AnyTraits<int64_t>())},
                                 {10, ATTR_DESC(group_type, AnyTraits<int64_t>())}};
ATTR_MAP(GroupedMatmul) = EMPTY_ATTR_MAP;
DYN_OUTPUT_MAP(GroupedMatmul) = {{0, DYN_OUTPUT_DESC(y)}};
REG_ADPT_DESC(GroupedMatmul, kNameGroupedMatmul, ADPT_DESC(GroupedMatmul))

// MoeInitRouting
INPUT_MAP(MoeInitRouting) = {{1, INPUT_DESC(x)}, {2, INPUT_DESC(row_idx)}, {3, INPUT_DESC(expert_idx)}};
ATTR_MAP(MoeInitRouting) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(MoeInitRouting) = {{4, ATTR_DESC(active_num, AnyTraits<int64_t>())}};
OUTPUT_MAP(MoeInitRouting) = {
  {0, OUTPUT_DESC(expanded_x)}, {1, OUTPUT_DESC(expanded_row_idx)}, {2, OUTPUT_DESC(expanded_expert_idx)}};
REG_ADPT_DESC(MoeInitRouting, kNameMoeInitRouting, ADPT_DESC(MoeInitRouting))

// MoeFinalizeRouting
INPUT_MAP(MoeFinalizeRouting) = {{1, INPUT_DESC(expanded_x)},
                                 {2, INPUT_DESC(x1)},
                                 {3, INPUT_DESC(x2)},
                                 {4, INPUT_DESC(bias)},
                                 {5, INPUT_DESC(scales)},
                                 {6, INPUT_DESC(expanded_row_idx)},
                                 {7, INPUT_DESC(expanded_expert_idx)}};
ATTR_MAP(MoeFinalizeRouting) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MoeFinalizeRouting) = {{0, OUTPUT_DESC(y)}};
REG_ADPT_DESC(MoeFinalizeRouting, kNameMoeFinalizeRouting, ADPT_DESC(MoeFinalizeRouting))

// MoeComputeExpertTokens
INPUT_MAP(MoeComputeExpertTokens) = {{1, INPUT_DESC(sorted_experts)}};
INPUT_ATTR_MAP(MoeComputeExpertTokens) = {{2, ATTR_DESC(num_experts, AnyTraits<int64_t>())}};
ATTR_MAP(MoeComputeExpertTokens) = EMPTY_ATTR_MAP;
OUTPUT_MAP(MoeComputeExpertTokens) = {{0, OUTPUT_DESC(total_rows_before_expert)}};
REG_ADPT_DESC(MoeComputeExpertTokens, kNameMoeComputeExpertTokens, ADPT_DESC(MoeComputeExpertTokens))

// GeGluV2
INPUT_MAP(GeGluV2) = {{1, INPUT_DESC(x)}};
ATTR_MAP(GeGluV2) = {{"dim", ATTR_DESC(dim, AnyTraits<int64_t>())},
                     {"approximate", ATTR_DESC(approximate, AnyTraits<int64_t>())},
                     {"activate_left", ATTR_DESC(activate_left, AnyTraits<bool>())}};
OUTPUT_MAP(GeGluV2) = {{0, OUTPUT_DESC(y)}, {1, OUTPUT_DESC(gelu)}};
REG_ADPT_DESC(GeGluV2, "GeGluV2", ADPT_DESC(GeGluV2))

// MoeGatingTopKSoftmax
INPUT_MAP(MoeGatingTopKSoftmax) = {{kIndex1, INPUT_DESC(x)}, {kIndex2, INPUT_DESC(finished)}};
ATTR_MAP(MoeGatingTopKSoftmax) = EMPTY_ATTR_MAP;
INPUT_ATTR_MAP(MoeGatingTopKSoftmax) = {{kIndex3, ATTR_DESC(k, AnyTraits<int64_t>())}};
OUTPUT_MAP(MoeGatingTopKSoftmax) = {
  {kIndex0, OUTPUT_DESC(y)}, {kIndex1, OUTPUT_DESC(expert_idx)}, {kIndex2, OUTPUT_DESC(row_idx)}};
REG_ADPT_DESC(MoeGatingTopKSoftmax, kNameMoeGatingTopKSoftmax, ADPT_DESC(MoeGatingTopKSoftmax))
}  // namespace mindspore::device::ascend
