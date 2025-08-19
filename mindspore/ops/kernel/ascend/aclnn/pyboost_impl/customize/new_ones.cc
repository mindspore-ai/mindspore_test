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

#include "kernel/ascend/aclnn/pyboost_impl/customize/new_ones.h"
#include "kernel/ascend/aclnn/pyboost_impl/auto_generate/ones.h"
#include "kernel/ascend/aclnn/pyboost_impl/aclnn_utils.h"
#include "plugin/ascend/res_manager/stream_manager/ascend_stream_manager.h"

namespace mindspore {
namespace kernel {
namespace pyboost {
tensor::TensorPtr NewOnesAscendCustomize(const std::shared_ptr<OpRunner> &op, const TensorPtr &input_tensor,
                                         const ValueTuplePtr &size, const std::optional<Int64ImmPtr> &dtype) {
  OpRunner::InferOpOutput(op, input_tensor, size, dtype);
  // No need to convert input
  PyBoostUtils::PrepareOpOutputs(op->device_context(), op->stream_id(), op->outputs());
  // Async
  PyBoostUtils::DispatchRun(std::make_shared<runtime::PyBoostDeviceTask>([op]() {
    MS_LOG(DEBUG) << "Run device task NewOnes start";
    auto device_context = op->device_context();
    const auto &outputs = op->outputs();
    // Malloc for output tensors
    PyBoostUtils::MallocOpOutputs(device_context, outputs);
    ScalarPtr fill_value = std::make_shared<Int64Imm>(1);
    LAUNCH_ACLNN(aclnnInplaceFillScalar, device_context, op->stream_id(), outputs[0], fill_value);
    MS_LOG(DEBUG) << "Run device task NewOnes end";
  }));
  return op->output(0);
}
}  // namespace pyboost
}  // namespace kernel
}  // namespace mindspore
