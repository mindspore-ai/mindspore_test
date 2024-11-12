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

#include "kernel/ascend/pyboost/customize/repeat_interleave_grad.h"
#include "plugin/device/ascend/hal/device/ascend_stream_manager.h"
#include "kernel/common/pyboost/op_register.h"
#include "kernel/common/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "kernel/common/pyboost/auto_generate/copy.h"
#include "kernel/common/pyboost/auto_generate/view.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::BaseTensorPtr RepeatInterleaveGradAscendCustomize(const std::shared_ptr<OpRunner> &op,
                                                          const BaseTensorPtr &input_tensor,
                                                          const BaseTensorPtr &repeats, const Int64ImmPtr &dim) {
  OpRunner::InferOpOutput(op, input_tensor, repeats, dim);
  const ShapeVector &output_shape = op->output_value_simple_info()->shape_vector_[0];
  auto repeats_shape = repeats->shape();
  if (repeats_shape.empty()) {
    repeats->set_shape(ShapeVector{1});
  }

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor, repeats);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  int64_t dim_imm = GetValue<int64_t>(dim);
  auto rank = SizeToLong(output_shape.size());
  dim_imm = (dim_imm < 0) ? (dim_imm + rank) : dim_imm;

  BaseTensorPtr input_tensor_contiguous = input_tensor;
  BaseTensorPtr repeats_contiguous = repeats;
  if (!input_tensor->is_contiguous()) {
    MS_LOG(DEBUG) << "For RepeatInterleaveGrad, input_tensor is not contiguous.";
    auto copy_op = CREATE_PYBOOST_OP(Copy, kAscendDevice);
    copy_op->set_stream_id(op->stream_id());
    input_tensor_contiguous = copy_op->Call(input_tensor);
  }
  if (!repeats->is_contiguous()) {
    MS_LOG(DEBUG) << "For RepeatInterleaveGrad, repeats is not contiguous.";
    auto copy_op = CREATE_PYBOOST_OP(Copy, kAscendDevice);
    copy_op->set_stream_id(op->stream_id());
    repeats_contiguous = copy_op->Call(repeats);
  }

  MS_LOG(DEBUG) << op->primitive()->name() << " Call start";
  // Async
  PyBoostUtils::DispatchRun(
    std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor_contiguous, repeats_contiguous, dim_imm]() {
      auto device_context = op->device_context();
      const auto &outputs = op->outputs();
      // Malloc for input tensors
      PyBoostUtils::MallocOpInputs(device_context, input_tensor_contiguous, repeats_contiguous);
      // Malloc for output tensors
      PyBoostUtils::MallocOpOutputs(device_context, outputs);

      LAUNCH_ACLNN(aclnnRepeatInterleaveGrad, device_context, op->stream_id(), input_tensor_contiguous,
                   repeats_contiguous, dim_imm, outputs[0]);
    }));

  MS_LOG(DEBUG) << op->primitive()->name() << " Launch end";
  return op->output(0);
}

}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
