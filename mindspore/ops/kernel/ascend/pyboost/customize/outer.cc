/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "kernel/ascend/pyboost/customize/outer.h"
#include "runtime/hardware/device_context_manager.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/auto_generate/reshape.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr OuterAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input,
                                       const TensorPtr &vec2) {
  OpRunner::InferOpOutput(op, input, vec2);

  auto reshape_op = CREATE_PYBOOST_OP(Reshape, op->device_context()->device_context_key_.device_name_);
  auto real_input = reshape_op->Call(input, std::make_shared<ValueTuple>(std::vector<ValuePtr>(
                                              {std::make_shared<Int64Imm>(-1), std::make_shared<Int64Imm>(1)})));

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), real_input, vec2);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, real_input, vec2]() {
    MS_LOG(DEBUG) << "Run device task Outer start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, real_input, vec2);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);

    LAUNCH_ACLNN(aclnnMul, device_context, op->stream_id(), real_input, vec2, outputs[0]);
    MS_LOG(DEBUG) << "Run device task Outer end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
