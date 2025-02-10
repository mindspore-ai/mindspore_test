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

#include "kernel/ascend/pyboost/customize/softmax.h"
#include <memory>
#include "plugin/res_manager/ascend/stream_manager/ascend_stream_manager.h"
#include "mindspore/ccsrc/pyboost/op_register.h"
#include "mindspore/ccsrc/pyboost/pyboost_utils.h"
#include "kernel/ascend/pyboost/aclnn_utils.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
namespace {
void SoftmaxAscendCall(const std::shared_ptr<OpRunner> &op, const device::DeviceContext *device_context,
                       const tensor::BaseTensorPtr &logits_tensor, const int64_t dim,
                       const std::vector<tensor::BaseTensorPtr> &outputs) {
  MS_LOG(DEBUG) << "Call start";
  LAUNCH_ACLNN(aclnnSoftmax, device_context, op->stream_id(), logits_tensor, dim, outputs[0]);
  MS_LOG(DEBUG) << "Launch end";
}
}  // namespace

tensor::BaseTensorPtr SoftmaxAscendCustomize(const std::shared_ptr<OpRunner> &op, const BaseTensorPtr &logits_tensor,
                                             const ValueTuplePtr &axis) {
  OpRunner::InferOpOutput(op, logits_tensor, axis);
  // ValueTuple to std::vector
  auto axis_vector = ConvertValueTupleToVector<int64_t>(axis);
  auto dim = axis_vector[0];

  PyBoostUtils::PrepareOpInputs(op->device_context(), op->stream_id(), logits_tensor);
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());

  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op, logits_tensor, dim]() {
    MS_LOG(DEBUG) << "Run device task Softmax start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for input tensors
    PyBoostUtils::MallocOpInputs(device_context, logits_tensor);
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    // Call aclnnSoftmax
    SoftmaxAscendCall(op, device_context, logits_tensor, dim, outputs);
    MS_LOG(DEBUG) << "Run device task Softmax end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
