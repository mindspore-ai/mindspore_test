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

#include "kernel/ascend/pyboost/customize/std.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr StdAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                     const std::optional<ValueTuplePtr> &dim, const Int64ImmPtr &correction,
                                     const BoolImmPtr &keepdim) {
  OpRunner::InferOpOutput(op, input_tensor, dim, correction, keepdim);
  std::vector<int64_t> dim_vector{};
  if (dim.has_value()) {
    dim_vector = ConvertValueTupleToVector<int64_t>(dim.value());
  }
  if (dim_vector.empty()) {
    dim_vector = GetRealDims(input_tensor->shape());
  }
  auto correction_imm = GetValue<int64_t>(correction);
  auto keepdim_imm = GetValue<bool>(keepdim);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, dim_vector, correction_imm, keepdim_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();

      PyBoostUtils::MallocOpInputs(device_context, input_tensor);
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
      LAUNCH_ACLNN(aclnnStd, device_context, op->stream_id(), input_tensor, dim_vector, correction_imm, keepdim_imm,
                   outputs[0]);
      MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
    }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
