/**
 * Copyright 2023-2024 Huawei Technologies Co., Ltd
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

#include "plugin/device/ascend/kernel/internal/flash_attention_score.h"

#include <memory>
#include <string>
#include "plugin/device/ascend/kernel/internal/internal_kernel_utils.h"
#include "plugin/device/ascend/kernel/internal/internal_kernel_in_out_map.h"
#include "mindspore/ops/op_def/op_enum.h"
#include "param/attention_param.h"
#include "utils/llm_manager.h"
#include "utils/ms_context.h"

namespace mindspore {
using mindspore::ops::FASInputLayoutMode;
namespace kernel {

constexpr size_t kInputFlashAttentionScoreQueryTHRank = 2;
constexpr size_t kAttnMaskLongSeqLen = 128;
constexpr size_t kAttnMaskDim2 = 2;

void InternalFlashAttentionScore::SetKVHead(internal::MixParam *op_param, int64_t head_num, int64_t input_layout) {
  if (input_layout == FASInputLayoutMode::TND) {
    op_param->kvHead = kv_shape_[kDim1];
  } else if (input_layout == FASInputLayoutMode::TH && q_shape_.size() == kInputFlashAttentionScoreQueryTHRank) {
    int64_t head_dim = q_shape_[kDim1] / head_num;
    op_param->kvHead = kv_shape_[kDim1] / head_dim;
  } else {
    int64_t head_dim = q_shape_[kDim2] / head_num;
    op_param->kvHead = kv_shape_[kDim2] / head_dim;

    if (enable_internal_fa_) {
      const size_t q_bnsd_size = 4;
      if (q_shape_.size() == q_bnsd_size) {
        head_dim = q_shape_[kDim3];
      }
      const int64_t head_dim_align = 16;
      if (head_dim % head_dim_align != 0) {
        MS_LOG(EXCEPTION) << kernel_name_ << ": 'head_dim' must be an integer multiple of 16 currently.";
      }
    }
  }
}

internal::OpParamPtr InternalFlashAttentionScore::CreateOpParam(const std::vector<KernelTensor *> &inputs,
                                                                const std::vector<KernelTensor *> &outputs) {
  auto param_ptr = std::make_shared<internal::FlashAttentionScoreParam>();
  // setup param from inputs
  q_shape_ = inputs[kIndex0]->GetShapeVector();
  kv_shape_ = inputs[kIndex1]->GetShapeVector();
  auto attn_mask_shape = inputs[kIndex6]->GetShapeVector();

  int64_t head_num = primitive_->HasAttr("head_num") ? GetValue<int64_t>(primitive_->GetAttr("head_num"))
                                                     : inputs[kIndex10]->GetValueWithCheck<int64_t>();

  int64_t pre_tokens = primitive_->HasAttr("pre_tokens") ? GetValue<int64_t>(primitive_->GetAttr("pre_tokens"))
                                                         : inputs[kIndex13]->GetValueWithCheck<int64_t>();
  int64_t next_tokens = primitive_->HasAttr("next_tokens") ? GetValue<int64_t>(primitive_->GetAttr("next_tokens"))
                                                           : inputs[kIndex14]->GetValueWithCheck<int64_t>();

  param_ptr->head_num = head_num;
  param_ptr->inner_precise = primitive_->HasAttr("inner_precise")
                               ? GetValue<int64_t>(primitive_->GetAttr("inner_precise"))
                               : inputs[kIndex15]->GetValueWithCheck<int64_t>();
  param_ptr->pre_tokens = pre_tokens;
  param_ptr->next_tokens = next_tokens;
  param_ptr->sparse_mode = primitive_->HasAttr("sparse_mode") ? GetValue<int64_t>(primitive_->GetAttr("sparse_mode"))
                                                              : inputs[kIndex17]->GetValueWithCheck<int64_t>();

  param_ptr->mask_dtype_ = InternalKernelUtils::ToInternalDType(inputs[kIndex6]->dtype_id());
  param_ptr->mask_dims_ = internal::VecToSVec<int64_t>(attn_mask_shape);

  int64_t input_layout = primitive_->HasAttr("input_layout") ? GetValue<int64_t>(primitive_->GetAttr("input_layout"))
                                                             : inputs[kIndex16]->GetValueWithCheck<int64_t>();

  internal::MixParam op_param;
  bool is_flatten_batch_seq = (input_layout == FASInputLayoutMode::TH || input_layout == FASInputLayoutMode::TND);
  if (soc_ == "ascend310p") {
    op_param.mixType = internal::MixParam::MixType::MIX_UNPAD_FLASH_ATTENTION_NZ_ENCODER_NOCACHE;
    op_param.maskType = internal::MixParam::MaskType::MASK_TYPE_NORM;
  } else {
    std::string high_precision_env = common::GetEnv("MS_INTERNAL_ENABLE_FA_HIGH_PRECISION");
    if (!high_precision_env.empty()) {
      op_param.mixType = internal::MixParam::MixType::MIX_UNPAD_FLASH_ATTENTION_FP32_ND;
    } else if (is_flatten_batch_seq) {
      op_param.mixType = internal::MixParam::MixType::MIX_UNPAD_FLASH_ATTENTION_ENCODER_ND;
    } else {
      op_param.mixType = internal::MixParam::MixType::MIX_UNPAD_DYNAMIC_BATCH_FLASH_ATTENTION;
    }
  }

  op_param.headSize = head_num;
  op_param.preTokens = pre_tokens;
  op_param.nextTokens = next_tokens;
  op_param.tor = primitive_->HasAttr("scale_value") ? GetValue<float>(primitive_->GetAttr("scale_value"))
                                                    : inputs[kIndex12]->GetValueWithCheck<float>();

  if (attn_mask_shape.size() == kAttnMaskDim2 && attn_mask_shape[kDim0] == kAttnMaskLongSeqLen) {
    op_param.isTriuMask = 1;
  }

  SetKVHead(&op_param, head_num, input_layout);
  if (is_flatten_batch_seq) {
    op_param.qSeqLen = ConvertActualSeqLengthsToVector(inputs[kIndex8]);
    op_param.kvSeqLen = ConvertActualSeqLengthsToVector(inputs[kIndex9]);
  } else {
    for (int64_t i = 0; i < kv_shape_[kDim0]; i++) {
      (void)op_param.qSeqLen.emplace_back(q_shape_[kDim1]);
      (void)op_param.kvSeqLen.emplace_back(q_shape_[kDim1]);
      (void)op_param.batchRunStatus.emplace_back(1);
    }
  }

  param_ptr->specificParam = op_param;
  param_ptr->opId = internal::OpId::FlashAttentionScore;
  return param_ptr;
}

bool InternalFlashAttentionScore::Init(const std::vector<KernelTensor *> &inputs,
                                       const std::vector<KernelTensor *> &outputs) {
  const std::string op_name = "FlashAttentionScore";
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto &enable_op_list = ms_context->ms_internal_enable_custom_kernel_list();
  enable_internal_fa_ = (std::find(enable_op_list.begin(), enable_op_list.end(), op_name) != enable_op_list.end());
  MS_LOG(INFO) << "Force op '" << kernel_name_ << "' to be resized to update op param 'seq_len'";
  return InternalKernelMod::Init(inputs, outputs);
}

MS_INTERNAL_KERNEL_FACTORY_REG(FlashAttentionScore, InternalFlashAttentionScore);
REG_MS_TO_INTERNAL_IN_TENSOR_IDX_MAP(FlashAttentionScore, INPUT_NUM_5, INDEX_0, INDEX_1, INDEX_2, INDEX_3, INDEX_6);
REG_MS_TO_INTERNAL_OUT_TENSOR_IDX_MAP(FlashAttentionScore, OUTPUT_NUM_1, INDEX_3);
}  // namespace kernel
}  // namespace mindspore
