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
#include "plugin/device/ascend/acl_ir/op_api_convert.h"
#include "plugin/res_manager/ascend/op_adapter/op_adapter_base.h"
#include "abstract/ops/primitive_infer_map.h"
#include "infer/ops_func_impl/fused_infer_attention_score.h"

namespace mindspore {
using mindspore::ops::FusedInferAttentionScoreInputIndex;
using mindspore::ops::FusedInferAttentionScoreOutputIndex;
namespace kernel {
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
                            << " for conversion to array must be of type Int32 or Int64, "
                            << "but got " << TypeIdToString(dtype_id);
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

void FusedInferAttentionScoreAscend::GetWorkSpaceInfo(const std::vector<KernelTensor *> &inputs,
                                                      const std::vector<KernelTensor *> &outputs) {
  // Convert to std::vector
  MS_EXCEPTION_IF_NULL(inputs[ops::kFusedInferAttentionScoreInputKeyIndex]);
  std::vector<KernelTensor *> key_vector{inputs[ops::kFusedInferAttentionScoreInputKeyIndex]};
  MS_EXCEPTION_IF_NULL(inputs[ops::kFusedInferAttentionScoreInputValueIndex]);
  std::vector<KernelTensor *> value_vector{inputs[ops::kFusedInferAttentionScoreInputValueIndex]};

  // Collect tensor elements to std::vector
  auto actual_seq_lengths_vector =
    ConvertTensorToVector("actual_seq_lengths", inputs[ops::kFusedInferAttentionScoreInputActualSeqLengthsIndex]);
  auto actual_seq_lengths_vector_pair = std::make_pair(actual_seq_lengths_vector, true);
  auto actual_seq_lengths_kv_vector =
    ConvertTensorToVector("actual_seq_lengths_kv", inputs[ops::kFusedInferAttentionScoreInputActualSeqLengthsKvIndex]);
  auto actual_seq_lengths_kv_vector_pair = std::make_pair(actual_seq_lengths_kv_vector, true);
  auto actual_shared_prefix_len_vector = ConvertTensorToVector(
    "actual_shared_prefix_len", inputs[ops::kFusedInferAttentionScoreInputActualSharedPrefixLenIndex]);
  auto actual_shared_prefix_len_vector_pair = std::make_pair(actual_shared_prefix_len_vector, true);

  // Convert to c++ scalar
  auto num_heads =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputNumHeadsIndex]);
  auto scale_value = device::ascend::ConvertKernelTensor<float>(inputs[ops::kFusedInferAttentionScoreInputScaleIndex]);
  // Note: aclnn requires the scale_value to be of type double. If a float value is passed, aclnn will internally
  // treat it as 0, which will result in incorrect computation.
  auto scale_value_d = static_cast<double>(scale_value);
  auto pre_tokens =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputPreTokensIndex]);
  auto next_tokens =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputNextTokensIndex]);
  auto input_layout =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputLayoutIndex]);
  auto input_layout_str = device::ascend::FASInputLayoutMode::ConvertEnumToString(input_layout);
  auto num_key_value_heads =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputNumKeyValueHeadsIndex]);
  auto sparse_mode =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputSparseModeIndex]);
  auto inner_precise =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputInnerPreciseIndex]);
  auto block_size =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputBlockSizeIndex]);
  auto antiquant_mode =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputAntiquantModeIndex]);
  auto softmax_lse_flag =
    device::ascend::ConvertKernelTensor<bool>(inputs[ops::kFusedInferAttentionScoreInputSoftmaxLseFlagIndex]);
  auto key_antiquant_mode =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputKeyAntiquantModeIndex]);
  auto value_antiquant_mode =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputValueAntiquantModeIndex]);

  GetWorkspaceForResize(
    inputs[ops::kFusedInferAttentionScoreInputQueryIndex], key_vector, value_vector,
    inputs[ops::kFusedInferAttentionScoreInputPseShiftIndex], inputs[ops::kFusedInferAttentionScoreInputAttnMaskIndex],
    actual_seq_lengths_vector_pair, actual_seq_lengths_kv_vector_pair,
    inputs[ops::kFusedInferAttentionScoreInputDequantScale1Index],
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
    inputs[ops::kFusedInferAttentionScoreInputValueSharedPrefixIndex], actual_shared_prefix_len_vector_pair, num_heads,
    scale_value_d, pre_tokens, next_tokens, input_layout_str, num_key_value_heads, sparse_mode, inner_precise,
    block_size, antiquant_mode, softmax_lse_flag, key_antiquant_mode, value_antiquant_mode,
    outputs[ops::kFusedInferAttentionScoreOutputAttentionOutIndex],
    outputs[ops::kFusedInferAttentionScoreOutputSoftmaxLseIndex]);
}

bool FusedInferAttentionScoreAscend::Launch(const std::vector<KernelTensor *> &inputs,
                                            const std::vector<KernelTensor *> &workspace,
                                            const std::vector<KernelTensor *> &outputs, void *stream_ptr) {
  MS_EXCEPTION_IF_NULL(stream_ptr);
  // Convert to std::vector
  MS_EXCEPTION_IF_NULL(inputs[ops::kFusedInferAttentionScoreInputKeyIndex]);
  std::vector<KernelTensor *> key_vector{inputs[ops::kFusedInferAttentionScoreInputKeyIndex]};
  MS_EXCEPTION_IF_NULL(inputs[ops::kFusedInferAttentionScoreInputValueIndex]);
  std::vector<KernelTensor *> value_vector{inputs[ops::kFusedInferAttentionScoreInputValueIndex]};

  // Collect tensor elements to std::vector
  auto actual_seq_lengths_vector =
    ConvertTensorToVector("actual_seq_lengths", inputs[ops::kFusedInferAttentionScoreInputActualSeqLengthsIndex]);
  auto actual_seq_lengths_vector_pair = std::make_pair(actual_seq_lengths_vector, true);
  auto actual_seq_lengths_kv_vector =
    ConvertTensorToVector("actual_seq_lengths_kv", inputs[ops::kFusedInferAttentionScoreInputActualSeqLengthsKvIndex]);
  auto actual_seq_lengths_kv_vector_pair = std::make_pair(actual_seq_lengths_kv_vector, true);
  auto actual_shared_prefix_len_vector = ConvertTensorToVector(
    "actual_shared_prefix_len", inputs[ops::kFusedInferAttentionScoreInputActualSharedPrefixLenIndex]);
  auto actual_shared_prefix_len_vector_pair = std::make_pair(actual_shared_prefix_len_vector, true);

  // Convert to c++ scalar
  auto num_heads =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputNumHeadsIndex]);
  auto scale_value = device::ascend::ConvertKernelTensor<float>(inputs[ops::kFusedInferAttentionScoreInputScaleIndex]);
  auto scale_value_d = static_cast<double>(scale_value);
  auto pre_tokens =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputPreTokensIndex]);
  auto next_tokens =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputNextTokensIndex]);
  auto input_layout =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputLayoutIndex]);
  auto input_layout_str = device::ascend::FASInputLayoutMode::ConvertEnumToString(input_layout);
  auto num_key_value_heads =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputNumKeyValueHeadsIndex]);
  auto sparse_mode =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputSparseModeIndex]);
  auto inner_precise =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputInnerPreciseIndex]);
  auto block_size =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputBlockSizeIndex]);
  auto antiquant_mode =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputAntiquantModeIndex]);
  auto softmax_lse_flag =
    device::ascend::ConvertKernelTensor<bool>(inputs[ops::kFusedInferAttentionScoreInputSoftmaxLseFlagIndex]);
  auto key_antiquant_mode =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputKeyAntiquantModeIndex]);
  auto value_antiquant_mode =
    device::ascend::ConvertKernelTensor<int64_t>(inputs[ops::kFusedInferAttentionScoreInputValueAntiquantModeIndex]);

  RunOp(stream_ptr, workspace, inputs[ops::kFusedInferAttentionScoreInputQueryIndex], key_vector, value_vector,
        inputs[ops::kFusedInferAttentionScoreInputPseShiftIndex],
        inputs[ops::kFusedInferAttentionScoreInputAttnMaskIndex], actual_seq_lengths_vector_pair,
        actual_seq_lengths_kv_vector_pair, inputs[ops::kFusedInferAttentionScoreInputDequantScale1Index],
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
        inputs[ops::kFusedInferAttentionScoreInputValueSharedPrefixIndex], actual_shared_prefix_len_vector_pair,
        num_heads, scale_value_d, pre_tokens, next_tokens, input_layout_str, num_key_value_heads, sparse_mode,
        inner_precise, block_size, antiquant_mode, softmax_lse_flag, key_antiquant_mode, value_antiquant_mode,
        outputs[ops::kFusedInferAttentionScoreOutputAttentionOutIndex],
        outputs[ops::kFusedInferAttentionScoreOutputSoftmaxLseIndex]);
  return true;
}

MS_ACLNN_KERNEL_FACTORY_REG(FusedInferAttentionScore, FusedInferAttentionScoreAscend);
}  // namespace kernel
}  // namespace mindspore
