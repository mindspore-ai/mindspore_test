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

#include "kernel/ascend/pyboost/customize/grouped_matmul_v2.h"
#include <memory>
#include <functional>
#include <vector>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
std::vector<TensorPtr> ConvertOptiaonlValueTupleToVector(const std::optional<ValueTuplePtr> &tensor_list_opt) {
  if (tensor_list_opt.has_value()) {
    return ConvertValueTupleToVector<TensorPtr>(tensor_list_opt.value());
  }
  return {};
}
}  // namespace
void GroupedMatmulV2AscendCustomize(const std::shared_ptr<OpRunner> &op, const ValueTuplePtr &x_tensor_list,
                                    const ValueTuplePtr &weight_tensor_list,
                                    const std::optional<ValueTuplePtr> &bias_tensor_list,
                                    const std::optional<ValueTuplePtr> &scale_tensor_list,
                                    const std::optional<ValueTuplePtr> &offset_tensor_list,
                                    const std::optional<ValueTuplePtr> &antiquant_scale_tensor_list,
                                    const std::optional<ValueTuplePtr> &antiquant_offset_tensor_list,
                                    const std::optional<ValueTuplePtr> &group_list, const Int64ImmPtr &split_item,
                                    const Int64ImmPtr &group_type) {
  MS_LOG(DEBUG) << "Call GroupedMatmul start";
  // Convert ValuePtr to c++ scalar
  OpRunner::InferOpOutput(op, x_tensor_list, weight_tensor_list, bias_tensor_list, scale_tensor_list,
                          offset_tensor_list, antiquant_scale_tensor_list, antiquant_offset_tensor_list, group_list,
                          split_item, group_type);

  std::vector<TensorPtr> x_tensor_list_vector = ConvertValueTupleToVector<TensorPtr>(x_tensor_list);
  std::vector<TensorPtr> weight_tensor_list_vector = ConvertValueTupleToVector<TensorPtr>(weight_tensor_list);
  std::vector<TensorPtr> bias_tensor_list_vector = ConvertOptiaonlValueTupleToVector(bias_tensor_list);
  std::vector<TensorPtr> scale_tensor_list_vector = ConvertOptiaonlValueTupleToVector(scale_tensor_list);
  std::vector<TensorPtr> offset_tensor_list_vector = ConvertOptiaonlValueTupleToVector(offset_tensor_list);
  std::vector<TensorPtr> antiquant_scale_tensor_list_vector =
    ConvertOptiaonlValueTupleToVector(antiquant_scale_tensor_list);
  std::vector<TensorPtr> antiquant_offset_tensor_list_vector =
    ConvertOptiaonlValueTupleToVector(antiquant_offset_tensor_list);

  std::vector<int64_t> group_list_real;
  if (group_list.has_value()) {
    group_list_real = ConvertValueTupleToVector<int64_t>(group_list.value());
  }
  auto split_item_imm = GetValue<int64_t>(split_item);
  auto group_type_imm = GetValue<int64_t>(group_type);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), x_tensor_list_vector, weight_tensor_list_vector,
                                bias_tensor_list_vector, scale_tensor_list_vector, offset_tensor_list_vector,
                                antiquant_scale_tensor_list_vector, antiquant_offset_tensor_list_vector);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>(
    [op, x_tensor_list_vector, weight_tensor_list_vector, bias_tensor_list_vector, scale_tensor_list_vector,
     offset_tensor_list_vector, antiquant_scale_tensor_list_vector, antiquant_offset_tensor_list_vector,
     group_list_real, split_item_imm, group_type_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, x_tensor_list_vector, weight_tensor_list_vector,
                                   bias_tensor_list_vector, scale_tensor_list_vector, offset_tensor_list_vector,
                                   antiquant_scale_tensor_list_vector, antiquant_offset_tensor_list_vector);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnGroupedMatmulV2, device_context, op->stream_id(), x_tensor_list_vector,
                   weight_tensor_list_vector, bias_tensor_list_vector, scale_tensor_list_vector,
                   offset_tensor_list_vector, antiquant_scale_tensor_list_vector, antiquant_offset_tensor_list_vector,
                   group_list_real, split_item_imm, group_type_imm, outputs);
      MS_LOG(DEBUG) << "Launch GroupedMatmul end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
