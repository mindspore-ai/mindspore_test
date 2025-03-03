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

#include "kernel/ascend/pyboost/customize/slice_ext.h"
#include <memory>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr SliceExtAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                          const Int64ImmPtr &dim_imm, const Int64ImmPtr &start_imm,
                                          const Int64ImmPtr &end_imm, const Int64ImmPtr &step_imm) {
  OpRunner::InferOpOutput(op, input_tensor, dim_imm, start_imm, end_imm, step_imm);

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), input_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  const auto &input_shape = input_tensor->shape();
  auto input_rank = SizeToLong(input_shape.size());

  auto dim = dim_imm->value();
  auto start = start_imm->value();
  auto end = end_imm->value();
  auto step = step_imm->value();

  dim = dim < 0 ? dim + input_rank : dim;

  auto dim_size = input_shape.at(dim);
  start = start < 0 ? start + dim_size : start;
  end = end < 0 ? end + dim_size : end;

  if (start < 0) {
    start = 0;
  } else if (start > dim_size) {
    start = dim_size;
  }

  if (end < start) {
    end = start;
  } else if (end > dim_size) {
    end = dim_size;
  }

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, input_tensor, dim, start, end, step]() {
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, input_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    // Launch aclnnSlice
    if (start != end) {
      LAUNCH_ACLNN(aclnnSlice, device_context, op->stream_id(), input_tensor, dim, start, end, step, outputs[0]);
    }
  }));

  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
