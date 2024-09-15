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

#include "kernel/ascend/pyboost/customize/norm.h"
#include <memory>
#include <functional>
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
constexpr size_t kNumberTwo = 2;
}  // namespace
void NormAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &input_x_tensor,
                         const std::optional<ScalarPtr> &ord, const std::optional<ValueTuplePtr> &dim,
                         const BoolImmPtr &keepdim, const std::optional<Int64ImmPtr> &dtype) {
  MS_LOG(DEBUG) << "Call Norm start";
  // Convert ValuePtr to c++ scalar
  OpRunner::InferOpOutput(op, input_x_tensor, ord, dim, keepdim, dtype);
  std::vector<int64_t> dim_vector{};
  if (dim.has_value()) {
    dim_vector = ConvertValueTupleToVector<int64_t>(dim.value());
  }
  ScalarPtr ord_scalar = nullptr;
  if (!ord.has_value()) {
    MAKE_SCALAR(kNumberTwo, kNumberTypeFloat32, ord_scalar);
  } else {
    ord_scalar = ord.value();
  }
  const auto keepdim_imm = GetValue<bool>(keepdim);
  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_x_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_x_tensor, ord_scalar, dim_vector, keepdim_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_x_tensor);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnNorm, device_context, op->stream_id(), input_x_tensor, ord_scalar, dim_vector, keepdim_imm,
                   outputs[kIndex0]);
      MS_LOG(DEBUG) << "Launch Norm end";
    }));
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
