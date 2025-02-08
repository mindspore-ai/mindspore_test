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

#include "kernel/ascend/pyboost/customize/grouped_matmul_v4.h"

#include <memory>
#include <functional>

#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<BaseTensorPtr> ConvertOptiaonlValueTupleToVector(const std::optional<ValueTuplePtr> &tensor_list_opt) {
  if (tensor_list_opt.has_value()) {
    return ConvertValueTupleToVector<BaseTensorPtr>(tensor_list_opt.value());
  }
  return {};
}
}  // namespace
void GroupedMatmulV4AscendCustomize(
  const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &x_tensor_list, const ValueTuplePtr &weight_tensor_list,
  const std::optional<ValueTuplePtr> &bias_tensor_list, const std::optional<ValueTuplePtr> &scale_tensor_list,
  const std::optional<ValueTuplePtr> &offset_tensor_list,
  const std::optional<ValueTuplePtr> &antiquant_scale_tensor_list,
  const std::optional<ValueTuplePtr> &antiquant_offset_tensor_list,
  const std::optional<ValueTuplePtr> &pre_token_scale_tensor_list, const std::optional<BaseTensorPtr> &group_list,
  const std::optional<ValueTuplePtr> &activation_input_tensor_list,
  const std::optional<ValueTuplePtr> &activation_quant_scale_tensor_list,
  const std::optional<ValueTuplePtr> &activation_quant_offset_tensor_list, const Int64ImmPtr &split_item_imm,
  const Int64ImmPtr &group_type_imm, const Int64ImmPtr &group_list_type_imm, const Int64ImmPtr &act_type_imm) {
  MS_LOG(DEBUG) << "Call GroupedMatmulExt start";
  OpRunner::InferOpOutput(op, x_tensor_list, weight_tensor_list, bias_tensor_list, scale_tensor_list,
                          offset_tensor_list, antiquant_scale_tensor_list, antiquant_offset_tensor_list,
                          pre_token_scale_tensor_list, group_list, activation_input_tensor_list,
                          activation_quant_scale_tensor_list, activation_quant_offset_tensor_list, split_item_imm,
                          group_type_imm, group_list_type_imm, act_type_imm);

  // Convert ValuePtr to c++ scalar
  std::vector<BaseTensorPtr> x = ConvertValueTupleToVector<BaseTensorPtr>(x_tensor_list);
  std::vector<BaseTensorPtr> weight = ConvertValueTupleToVector<BaseTensorPtr>(weight_tensor_list);
  std::vector<BaseTensorPtr> bias = ConvertOptiaonlValueTupleToVector(bias_tensor_list);
  std::vector<BaseTensorPtr> scale = ConvertOptiaonlValueTupleToVector(scale_tensor_list);
  std::vector<BaseTensorPtr> offset = ConvertOptiaonlValueTupleToVector(offset_tensor_list);
  std::vector<BaseTensorPtr> antiquant_scale = ConvertOptiaonlValueTupleToVector(antiquant_scale_tensor_list);
  std::vector<BaseTensorPtr> antiquant_offset = ConvertOptiaonlValueTupleToVector(antiquant_offset_tensor_list);
  std::vector<BaseTensorPtr> pre_token_scale = ConvertOptiaonlValueTupleToVector(pre_token_scale_tensor_list);
  std::vector<BaseTensorPtr> activation_input = ConvertOptiaonlValueTupleToVector(activation_input_tensor_list);
  std::vector<BaseTensorPtr> activation_quant_scale =
    ConvertOptiaonlValueTupleToVector(activation_quant_scale_tensor_list);
  std::vector<BaseTensorPtr> activation_quant_offset =
    ConvertOptiaonlValueTupleToVector(activation_quant_offset_tensor_list);

  auto split_item = GetValue<int64_t>(split_item_imm);
  auto group_type = GetValue<int64_t>(group_type_imm);
  auto group_list_type = GetValue<int64_t>(group_list_type_imm);
  auto act_type = GetValue<int64_t>(act_type_imm);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x, weight, bias, scale, offset, antiquant_scale,
                                antiquant_offset, pre_token_scale, group_list, activation_input, activation_quant_scale,
                                activation_quant_offset);

  std::vector<BaseTensorPtr> activation_feature_out;
  std::vector<BaseTensorPtr> dyn_quant_scale_out;
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x, weight, bias, scale, offset, antiquant_scale, antiquant_offset, pre_token_scale, group_list,
     activation_input, activation_quant_scale, activation_quant_offset, split_item, group_type, group_list_type,
     act_type, activation_feature_out, dyn_quant_scale_out]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x, weight, bias, scale, offset, antiquant_scale, antiquant_offset,
                                   pre_token_scale, group_list, activation_input, activation_quant_scale,
                                   activation_quant_offset);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnGroupedMatmulV4, device_context, op->stream_id(), x, weight, bias, scale, offset,
                   antiquant_scale, antiquant_offset, pre_token_scale, group_list, activation_input,
                   activation_quant_scale, activation_quant_offset, split_item, group_type, group_list_type, act_type,
                   outputs, activation_feature_out, dyn_quant_scale_out);
      MS_LOG(DEBUG) << "Launch GroupedMatmulExt end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
