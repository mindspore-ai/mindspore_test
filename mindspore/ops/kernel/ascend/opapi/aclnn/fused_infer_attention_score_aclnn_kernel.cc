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
#include "kernel/ascend/opapi/aclnn/fused_infer_attention_score_aclnn_kernel.h"
#include <algorithm>
#include <vector>
#include <memory>
#include <functional>
#include "ir/tensor.h"
#include "runtime/device/kernel_runtime.h"
#include "kernel/ascend/acl_ir/op_api_convert.h"
#include "plugin/res_manager/ascend/op_adapter/op_adapter_base.h"
#include "abstract/ops/primitive_infer_map.h"
#include "infer/ops_func_impl/fused_infer_attention_score.h"
#include "utils/llm_manager.h"

namespace mindspore {
using mindspore::ops::FusedInferAttentionScoreInputIndex;
using mindspore::ops::FusedInferAttentionScoreOutputIndex;
namespace kernel {
namespace fused_infer_attention_score {
std::vector<int64_t> FusedInferAttentionScoreAscend::ConvertTensorToVector(const std::string &tensor_name,
                                                                           KernelTensor *const tensor_ptr) {
  MS_EXCEPTION_IF_NULL(tensor_ptr);
  std::vector<int64_t> res;
  if (tensor_ptr->type_id() == kMetaTypeNone) {
    return res;
  }
  // check tensor shape
  MS_EXCEPTION_IF_NULL(primitive_);
  auto shape = tensor_ptr->GetShapeVector();
  if (shape.size() != 1) {
    auto print_shape = [](const ShapeVector &shape) -> std::string {
      std::ostringstream oss;
      oss << "[";
      for (size_t i = 0; i < shape.size(); i++) {
        if (i != 0) {
          oss << ", " << shape[i];
        } else {
          oss << shape[i];
        }
      }
      oss << "]";
      return oss.str();
    };
    MS_EXCEPTION(TypeError) << "For " << primitive_->name() << ", the input " << tensor_name
                            << " for conversion to array must be 1 dimension, but got " << shape.size()
                            << " dimension with shape " << print_shape(shape);
  }
  // check tensor type
  TypeId dtype_id = tensor_ptr->dtype_id();
  if (dtype_id != TypeId::kNumberTypeInt64 && dtype_id != TypeId::kNumberTypeInt32) {
    MS_EXCEPTION(TypeError) << "For " << primitive_->name() << ", the input " << tensor_name
                            << " for conversion to array must be of type Int32 or Int64, but got "
                            << TypeIdToString(dtype_id);
  }
  // collect tensor elements to vector
  if (dtype_id == kNumberTypeInt64) {
    res = tensor_ptr->GetValueWithCheck<std::vector<int64_t>>();
  } else {
    std::vector<int32_t> vector_int32 = tensor_ptr->GetValueWithCheck<std::vector<int32_t>>();
    res.assign(vector_int32.begin(), vector_int32.end());
  }
  return res;
}

void FusedInferAttentionScoreAscend::SetScalarParam(const std::vector<KernelTensor *> &inputs) {
  num_heads_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputNumHeadsIndex]);
  auto scale_value = device::ascend::ConvertKernelTensor<float>(inputs[ops::kFusedInferAttentionScoreInputScaleIndex]);
  // Note: aclnn requires the scale_value to be of type double. If a float value is passed, aclnn will internally
  // treat it as 0, which will result in incorrect computation.
  scale_value_d_ = static_cast<double>(scale_value);
  pre_tokens_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputPreTokensIndex]);
  next_tokens_ =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputNextTokensIndex]);
  auto input_layout =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputLayoutIndex]);
  input_layout_str_ = device::ascend::FASInputLayoutMode::ConvertEnumToString(input_layout);
  num_key_value_heads_ =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputNumKeyValueHeadsIndex]);
  sparse_mode_ =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputSparseModeIndex]);
  inner_precise_ =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputInnerPreciseIndex]);
  block_size_ = device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputBlockSizeIndex]);
  antiquant_mode_ =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputAntiquantModeIndex]);
  softmax_lse_flag_ =
    device::ascend::ConvertKernelTensor<bool>(inputs[ops::kFusedInferAttentionScoreInputSoftmaxLseFlagIndex]);
  key_antiquant_mode_ =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputKeyAntiquantModeIndex]);
  value_antiquant_mode_ =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputValueAntiquantModeIndex]);

  return;
}

bool FusedInferAttentionScoreAscend::Init(const std::vector<KernelTensor *> &inputs,
                                          const std::vector<KernelTensor *> &outputs) {
  auto &llm_manager = LLMManager::GetInstance();
  llm_manager.add_force_resize_kernel(kernel_name_);

  SetScalarParam(inputs);
  return true;
}

void FusedInferAttentionScoreAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  SetScalarParam(inputs);
  // Convert to std::vector
  MS_EXCEPTION_IF_NULL(inputs[ops::kFusedInferAttentionScoreInputKeyIndex]);
  key_vector_ = {inputs[ops::kFusedInferAttentionScoreInputKeyIndex]};
  MS_EXCEPTION_IF_NULL(inputs[ops::kFusedInferAttentionScoreInputValueIndex]);
  value_vector_ = {inputs[ops::kFusedInferAttentionScoreInputValueIndex]};

  std::vector<int64_t> actual_q_lengths_vector = {};
  std::vector<int64_t> actual_kv_lengths_vector = {};
  std::vector<int64_t> actual_shared_prefix_lengths_vector = {};
  auto &llm_manager = LLMManager::GetInstance();
  auto ret = llm_manager.GetGraphInputToVector<int32_t, int64_t>("q_seq_lens", &actual_q_lengths_vector);
  if (!ret) {
    actual_q_lengths_vector =
      ConvertTensorToVector("actual_seq_lengths", inputs[ops::kFusedInferAttentionScoreInputActualSeqLengthsIndex]);
  }
  if (actual_q_lengths_vector.size() == 1 && actual_q_lengths_vector[0] == 1) {
    // actual_q_lengths_vector is best left empty because it affects the execution time of GetWorkspaceForResize
    actual_q_lengths_vector.clear();
  }
  ret = llm_manager.GetGraphInputToVector<int32_t, int64_t>("batch_valid_length", &actual_kv_lengths_vector);
  if (!ret) {
    actual_kv_lengths_vector = ConvertTensorToVector(
      "actual_seq_lengths_kv", inputs[ops::kFusedInferAttentionScoreInputActualSeqLengthsKvIndex]);
  }
  ret = llm_manager.GetGraphInputToVector<int32_t, int64_t>("actual_shared_prefix_len",
                                                            &actual_shared_prefix_lengths_vector);
  if (!ret) {
    actual_shared_prefix_lengths_vector = ConvertTensorToVector(
      "actual_shared_prefix_len", inputs[ops::kFusedInferAttentionScoreInputActualSharedPrefixLenIndex]);
  }

  actual_q_lengths_vector_pair_ = std::make_pair(actual_q_lengths_vector, true);
  actual_kv_lengths_vector_pair_ = std::make_pair(actual_kv_lengths_vector, true);
  actual_shared_prefix_lengths_vector_pair_ = std::make_pair(actual_shared_prefix_lengths_vector, true);

  GetWorkspaceForResize(inputs[ops::kFusedInferAttentionScoreInputQueryIndex], key_vector_, value_vector_,
                        inputs[ops::kFusedInferAttentionScoreInputPseShiftIndex],
                        inputs[ops::kFusedInferAttentionScoreInputAttnMaskIndex], actual_q_lengths_vector_pair_,
                        actual_kv_lengths_vector_pair_, inputs[ops::kFusedInferAttentionScoreInputDequantScale1Index],
                        inputs[ops::kFusedInferAttentionScoreInputQuantScale1Index],
                        inputs[ops::kFusedInferAttentionScoreInputDequantScale2Index],
                        inputs[ops::kFusedInferAttentionScoreInputQuantScale2Index],
                        inputs[ops::kFusedInferAttentionScoreInputQuantOffset2Index],
                        inputs[ops::kFusedInferAttentionScoreInputAntiquantScaleIndex],
                        inputs[ops::kFusedInferAttentionScoreInputAntiquantOffsetIndex],
                        inputs[ops::kFusedInferAttentionScoreInputBlockTableIndex],
                        inputs[ops::kFusedInferAttentionScoreInputQueryPaddingSizeIndex],
                        inputs[ops::kFusedInferAttentionScoreInputKvPaddingSizeIndex],
                        inputs[ops::kFusedInferAttentionScoreInputKeyAntiquantScaleIndex],
                        inputs[ops::kFusedInferAttentionScoreInputKeyAntiquantOffsetIndex],
                        inputs[ops::kFusedInferAttentionScoreInputValueAntiquantScaleIndex],
                        inputs[ops::kFusedInferAttentionScoreInputValueAntiquantOffsetIndex],
                        inputs[ops::kFusedInferAttentionScoreInputKeySharedPrefixIndex],
                        inputs[ops::kFusedInferAttentionScoreInputValueSharedPrefixIndex],
                        actual_shared_prefix_lengths_vector_pair_, num_heads_, scale_value_d_, pre_tokens_,
                        next_tokens_, input_layout_str_, num_key_value_heads_, sparse_mode_, inner_precise_,
                        block_size_, antiquant_mode_, softmax_lse_flag_, key_antiquant_mode_, value_antiquant_mode_,
                        outputs[ops::kFusedInferAttentionScoreOutputAttentionOutIndex],
                        outputs[ops::kFusedInferAttentionScoreOutputSoftmaxLseIndex]);
}

bool FusedInferAttentionScoreAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);

  RunOp(stream_ptr, workspace, inputs[ops::kFusedInferAttentionScoreInputQueryIndex], key_vector_, value_vector_,
        inputs[ops::kFusedInferAttentionScoreInputPseShiftIndex],
        inputs[ops::kFusedInferAttentionScoreInputAttnMaskIndex], actual_q_lengths_vector_pair_,
        actual_kv_lengths_vector_pair_, inputs[ops::kFusedInferAttentionScoreInputDequantScale1Index],
        inputs[ops::kFusedInferAttentionScoreInputQuantScale1Index],
        inputs[ops::kFusedInferAttentionScoreInputDequantScale2Index],
        inputs[ops::kFusedInferAttentionScoreInputQuantScale2Index],
        inputs[ops::kFusedInferAttentionScoreInputQuantOffset2Index],
        inputs[ops::kFusedInferAttentionScoreInputAntiquantScaleIndex],
        inputs[ops::kFusedInferAttentionScoreInputAntiquantOffsetIndex],
        inputs[ops::kFusedInferAttentionScoreInputBlockTableIndex],
        inputs[ops::kFusedInferAttentionScoreInputQueryPaddingSizeIndex],
        inputs[ops::kFusedInferAttentionScoreInputKvPaddingSizeIndex],
        inputs[ops::kFusedInferAttentionScoreInputKeyAntiquantScaleIndex],
        inputs[ops::kFusedInferAttentionScoreInputKeyAntiquantOffsetIndex],
        inputs[ops::kFusedInferAttentionScoreInputValueAntiquantScaleIndex],
        inputs[ops::kFusedInferAttentionScoreInputValueAntiquantOffsetIndex],
        inputs[ops::kFusedInferAttentionScoreInputKeySharedPrefixIndex],
        inputs[ops::kFusedInferAttentionScoreInputValueSharedPrefixIndex], actual_shared_prefix_lengths_vector_pair_,
        num_heads_, scale_value_d_, pre_tokens_, next_tokens_, input_layout_str_, num_key_value_heads_, sparse_mode_,
        inner_precise_, block_size_, antiquant_mode_, softmax_lse_flag_, key_antiquant_mode_, value_antiquant_mode_,
        outputs[ops::kFusedInferAttentionScoreOutputAttentionOutIndex],
        outputs[ops::kFusedInferAttentionScoreOutputSoftmaxLseIndex]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(FusedInferAttentionScore, FusedInferAttentionScoreAscend);
}  // namespace fused_infer_attention_score
}  // namespace kernel
}  // namespace mindspore
