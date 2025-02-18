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

#include "kernel/ascend/pyboost/customize/speed_fusion_attention_grad.h"
#include "kernel/ascend/pyboost/auto_generate/dropout_gen_mask_ext.h"
#include "kernel/ascend/pyboost/auto_generate/zeros.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "runtime/device/device_address_utils.h"
#include "plugin/res_manager/ascend/op_adapter/op_adapter_base.h"
#include "mindspore/ops/op_def/op_enum.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
void SpeedFusionAttentionGradAscendCustomize(
  const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &query, const BaseTensorPtr &key, const BaseTensorPtr &value,
  const BaseTensorPtr &dy, const Int64ImmPtr &head_num, const Int64ImmPtr &input_layout,
  const std::optional<BaseTensorPtr> &pse, const std::optional<BaseTensorPtr> &padding_mask,
  const std::optional<BaseTensorPtr> &atten_mask, const std::optional<BaseTensorPtr> &softmax_max,
  const std::optional<BaseTensorPtr> &softmax_sum, const std::optional<BaseTensorPtr> &softmax_in,
  const std::optional<BaseTensorPtr> &attention_in, const FP32ImmPtr &scale, const FP32ImmPtr &keep_prob,
  const Int64ImmPtr &pre_tokens, const Int64ImmPtr &next_tokens, const Int64ImmPtr &inner_precise,
  const std::optional<BaseTensorPtr> &seed, const std::optional<BaseTensorPtr> &offset,
  const std::optional<BaseTensorPtr> &numels, const std::optional<ValueTuplePtr> &prefix,
  const std::optional<ValueTuplePtr> &actual_seq_qlen, const std::optional<ValueTuplePtr> &actual_seq_kvlen,
  const Int64ImmPtr &sparse_mode, const BoolImmPtr &gen_mask_parallel, const BoolImmPtr &sync,
  const Int64ImmPtr &pse_type, const std::optional<ValueTuplePtr> &q_start_idx,
  const std::optional<ValueTuplePtr> &kv_start_idx) {
  OpRunner::InferOpOutput(op, query, key, value, dy, head_num, input_layout, pse, padding_mask, atten_mask, softmax_max,
                          softmax_sum, softmax_in, attention_in, scale, keep_prob, pre_tokens, next_tokens,
                          inner_precise, seed, offset, numels, prefix, actual_seq_qlen, actual_seq_kvlen, sparse_mode,
                          gen_mask_parallel, sync, pse_type, q_start_idx, kv_start_idx);

  if (!seed.has_value() || !offset.has_value() || !numels.has_value()) {
    MS_EXCEPTION(ValueError) << "For " << op->primitive()->name() << ", seed, offset and numels must have value.";
  }

  std::optional<BaseTensorPtr> dropout_mask = std::nullopt;
  auto numels_tensor = numels.value();
  numels_tensor->data_sync();
  int64_t numels_value = *static_cast<int64_t *>(numels_tensor->data_c());
  double keep_prob_value = static_cast<double>(GetValue<float>(keep_prob));
  if (0 < keep_prob_value && keep_prob_value < 1.) {
    auto p = std::make_shared<FP32Imm>(static_cast<float>(1 - keep_prob_value));
    auto shape = std::make_shared<ValueTuple>(std::vector<ValuePtr>{MakeValue(numels_value)});
    auto dtype = std::make_shared<Int64Imm>(static_cast<int64_t>(query->Dtype()->type_id()));
    auto dropout_gen_mask_ext_op =
      CREATE_PYBOOST_OP(DropoutGenMaskExt, op->device_context()->device_context_key_.device_name_);
    dropout_mask.emplace(dropout_gen_mask_ext_op->Call(shape, p, seed.value(), offset.value(), dtype));
  } else if (keep_prob_value == 0) {
    constexpr int64_t kAlignBitNum = 128;
    constexpr int64_t kBitOfByte = 8;
    int64_t align_length = (numels_value + kAlignBitNum - 1) / kAlignBitNum * kAlignBitNum / kBitOfByte;
    auto shape = std::make_shared<ValueTuple>(std::vector<ValuePtr>{MakeValue(align_length)});
    auto zeros_op = CREATE_PYBOOST_OP(Zeros, op->device_context()->device_context_key_.device_name_);
    dropout_mask.emplace(zeros_op->Call(shape, std::make_shared<Int64Imm>(static_cast<int64_t>(kNumberTypeUInt8))));
  }

  auto head_num_value = GetValue<int64_t>(head_num);
  auto input_layout_str =
    mindspore::device::ascend::FASInputLayoutMode::ConvertEnumToString(GetValue<int64_t>(input_layout));
  auto scale_value = static_cast<double>(GetValue<float>(scale));
  auto pre_tokens_value = GetValue<int64_t>(pre_tokens);
  auto next_tokens_value = GetValue<int64_t>(next_tokens);
  auto inner_precise_value = GetValue<int64_t>(inner_precise);
  auto sparse_mode_value = GetValue<int64_t>(sparse_mode);
  auto pse_type_value = GetValue<int64_t>(pse_type);
  auto prefix_array = ConvertValueTupleToVector<int64_t>(prefix);
  auto actual_seq_qlen_array = ConvertValueTupleToVector<int64_t>(actual_seq_qlen);
  auto actual_seq_kvlen_array = ConvertValueTupleToVector<int64_t>(actual_seq_kvlen);
  std::optional<std::vector<int64_t>> q_start_idx_array = std::nullopt;
  std::optional<std::vector<int64_t>> kv_start_idx_array = std::nullopt;
  if (q_start_idx.has_value()) {
    q_start_idx_array.emplace(ConvertValueTupleToVector<int64_t>(q_start_idx.value()));
  }
  if (kv_start_idx.has_value()) {
    kv_start_idx_array.emplace(ConvertValueTupleToVector<int64_t>(kv_start_idx.value()));
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), query, key, value, dy, pse, dropout_mask,
                                padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, query, key, value, dy, head_num_value, input_layout_str, pse, dropout_mask, padding_mask, atten_mask,
     softmax_max, softmax_sum, softmax_in, attention_in, scale_value, keep_prob_value, pre_tokens_value,
     next_tokens_value, inner_precise_value, prefix_array, actual_seq_qlen_array, actual_seq_kvlen_array,
     sparse_mode_value, pse_type_value, q_start_idx_array, kv_start_idx_array]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, query, key, value, dy, pse, dropout_mask, padding_mask, atten_mask,
                                   softmax_max, softmax_sum, softmax_in, attention_in);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);
      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      if (!actual_seq_qlen_array.empty() && !actual_seq_kvlen_array.empty()) {
        LAUNCH_ACLNN(aclnnFlashAttentionUnpaddingScoreGradV2, device_context, op->stream_id(), query, key, value, dy,
                     pse, dropout_mask, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in,
                     prefix_array, actual_seq_qlen_array, actual_seq_kvlen_array, q_start_idx_array, kv_start_idx_array,
                     scale_value, keep_prob_value, pre_tokens_value, next_tokens_value, head_num_value,
                     input_layout_str, inner_precise_value, sparse_mode_value, pse_type_value, outputs[kIndex0],
                     outputs[kIndex1], outputs[kIndex2], outputs[kIndex3]);
      } else {
        LAUNCH_ACLNN(aclnnFlashAttentionScoreGradV2, device_context, op->stream_id(), query, key, value, dy, pse,
                     dropout_mask, padding_mask, atten_mask, softmax_max, softmax_sum, softmax_in, attention_in,
                     prefix_array, q_start_idx_array, kv_start_idx_array, scale_value, keep_prob_value,
                     pre_tokens_value, next_tokens_value, head_num_value, input_layout_str, inner_precise_value,
                     sparse_mode_value, pse_type_value, outputs[kIndex0], outputs[kIndex1], outputs[kIndex2],
                     outputs[kIndex3]);
      }
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
